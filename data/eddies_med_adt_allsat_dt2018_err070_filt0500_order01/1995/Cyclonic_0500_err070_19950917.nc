CDF       
      obs    H   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��+J        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�V�   max       P��        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��   max       <�o        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?(�\)   max       @E��
=p�     @  !   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @v�z�G�     @  ,L   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1         max       @P�           �  7�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ˊ        max       @��             8   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��P   max       <T��        9<   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��J   max       B.�        :\   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�x�   max       B.@�        ;|   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�y�   max       C�ւ        <�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�M@   max       C��C        =�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          c        >�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          K        ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;        A   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�V�   max       P��        B<   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�-V�   max       ?�{���m]        C\   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��   max       <�o        D|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?(�\)   max       @E�\(��     @  E�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v
=p��     @  P�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @P�           �  \   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ˊ        max       @�'`            \�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A7   max         A7        ]�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�N;�5�Y   max       ?޳g��	l     �  ^�               b               
                $         %   (                              4   *   !   %   &                           !            8         =                        G            %   ;         &      (               N��NՆ�N3qOO8˲P��N��N3$qN���Ol`IN��O�ZM�x�O@�wN���P/�NFƧO�cfO��P��N0sO�~�O�q�Ny�Oga�NPa�N ��O,	{O��iP-��O�D�O���O���O���M�V�O�N�1�N��#N���M
N�N1�Oj�*O0��O���N4��O�|�OD�qN�O���O���N��O,��N�[�N��aO��O�O�΃Nyr�O=��N�v.O���O�!mNBdPO��fO��@OW�Of�O!��O:^N�g�O2I�NQо<�o;�`B;ě�;��
;�o:�o��o�o�D����o��o��`B��`B�o�#�
�49X�D����o��t���1�ě����ͼ��ͼ�����h���o�+�+�C��C��\)�\)�t��t���P��w�,1�,1�0 Ž0 Ž8Q�8Q�8Q�8Q�H�9�H�9�H�9�T���T���T���T���T���T���aG��e`B��%��+��+��+��hs��hs���㽝�-���-���-���
���"ѽ�"ѽ�`B����������������������:;CHLT_akdaaTLHE=;::��������������������##<HUan}|zrnaUHD<4,#0U{������gK<������������������������������������������������������������������������������������������������������*6BO[\\fie[OB6)��


������������"/9<@BD@</#��������������������36BP[g���������g[B53Y[ahtzztthf[YYYYYYYY��
#=DB<#
����������
 #���������1@[t�������gXNB5+)+1����������������������������������������im������������zmfcdi������������������������������������������������������������������������������������������1/03:HTadca]TMH;/,,1oz�������������oorno)6BOht�����ta[O>.#!)@BIO[t�����th[OB=:<@������������������������������������eht���xthgeeeeeeeeee��������������������%/<HPUVUTOH?<3//%%%%�������

���������?BKOZ[^_a[ZOB?;:????����������������������
 
������������"#/7<C=<9/$#""""""""����������������~�������� �������r{���������������~sr&)688A6)'&&&&&&&&&&&������������'-8<IMMMOJC<0#SU`bn{|{rnbWUQSSSSSS@O[ht�����th[B2%")@��)5)$������?BN[gitvvtg[NB57????)05676()67:986-)'"X^dhmt�����wtih[WVX58:BNRgt���tg[NB6015��������������������+<HUw�|pe^UHEB3.**+�������������������������������������������� ��������MXgt�����������t[OLMkpz������������znihk����������������������������������������������������������������#/6<CEGF</-#it{���������������ti)35BENXNJB55)"bghpt���������tgbbbb���������}zbWRUaknw�nronca_UUUV[annnnnnn��������������������������������������������������
��#�$�&�#���
������Ŀ������������ĿͿѿѿѿĿĿĿĿĿĿĿ����������������������������������������˽��p�l�s�������н����A�Z�w�s�M��ݽ�������
�����!�-�:�C�B�:�9�-�!���ÇÆ�~ÆÇÓØàâàÓÇÇÇÇÇÇÇÇÇ�����ܻлȻлӻܻ����������������������¿�����������
����
�������¦­²¶¸²¦�������������������Ŀѿڿ޿ֿܿпͿĿ����z�x�y�zÇÓÖÓÑÇ�z�z�z�z�z�z�z�z�z�z�<�9�/�#�/�5�<�H�U�a�e�p�{�z�x�n�a�U�H�<�H�@�<�5�3�<�H�U�`�a�c�b�a�\�Y�U�H�H�H�H��ݿп������������ѿݿ�������������/�,�#���#�/�<�C�>�<�2�/�/�/�/�/�/�/�/����������)�6�B�T�\�g�V�B�7�6�)���r�j�q�y����������ʼμڼؼּ���������r���z�a�]�a�z����������������������������àÙÖÛàéìôðìàààààààààà�l�S�F�3�+����#�-�:�S�l�x���������x�l����������Ƽ��������� �%�%�#��������׾ҾԾ׾ؾ��������׾׾׾׾׾׾׾��H�>�;�2�-�$�"��"�-�;�H�T�a�f�h�h�a�T�H������������!�"�!�����������������m�e�`�T�L�T�`�m�y�z�y�p�m�m�m�m�m�m�m�m�`�P�I�<�7�<�I�M�U�b�n�r�{ŃŇŌŋ�{�n�`�g�Z�N�A�5�*�(�-�5�A�O�Z�}�������������g�H�<�6�9�D�T�a�z�������������������z�T�H�b�j�l�u�����������������������������s�b�/�"������
�"�/�H�T�_�c�e�c�_�]�T�H�;�/��ĿīĦĞĦĴĿ��������(�$���������̺~�r�|�����������ɺֺݺ����ʺ������~�r�p�q�q�r�t�~����~�r�r�r�r�r�r�r�r�r�r�~�w�n�y�����Ǻɺ����ֺɺ����������~����������������������������������������D�D�D�D�D�D�D�D�D�D�D�EEEEEE D�D�D컪���������������ûλлػԻлû����������������������������������������������������������������������������������������������������������������������������������B�?�<�?�B�H�O�[�h�rāĈĈą�z�t�h�[�O�B�[�Y�O�J�I�O�[�b�h�t�}āčėčā�t�h�b�[�b�V�I�?�=�I�O�W�[�^�b�o�{ǀǉǌǈ�{�o�b���������������������������������������Ҽ4�'�"����'�4�M�Y�r�������������Y�4�������������������������������������������|�s�s�k�q�s�v�����������������������������������ŹϹܹ����� ������ܹϹù�����ھ�����	����"�'�7�@�>�.�"���������������	������	�������ּμʼ��������ʼּݼ������������ּr�g�n�r�������������������r�r�r�r�r�r�����������z�����������������ĽŽȽĽ����
�����������'�C�Q�[�Y�M�I�H�C�6�*��
�m�h�`�b�e�h�m�y�������������������y�m�m���������������ʾ���������׾ʾ�����Ň�{ŇŌŔŠŭűŹźŹŭŠŔŇŇŇŇŇŇ�r�g�k�u������������ʼ̼ʼ�����������r�ּѼּټ�����������������ۼ���ĿĴįīĦįĪĳĿ�������������������̹�ܹԹȹǹܹ���3�@�L�V�]�[�S�B�'��������������������������������������²�s�q�u¦��������������¿²�G�=�-�(�/�:�G�S�`���������������l�`�S�G�=�4������ؽнĽݽ���4�0�4�A�K�G�=F$F FFFFFF$F1F=FJFSFLFJFCF=F2F1F$F$�#�!���
�����������
��#�-�/�9�<�*�%�#àÙ×àâëìóùÿ������������üùìàùòìàÝÙÕØàìöùýýýùùùùù���������ĿѿٿٿտѿϿĿ���������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� 8 = U _ 4 9 D F 1 7 < 6 0 [ 5 ? Z B L W e $ q : ` X 4 l . U 0 \ a � � C H + � x a " 3 V U : C � Q B b 0 O T j Q : [ Y ] P P D P 9 r 8 7 G Z � /  (  �  Q  �  �    \  (  �  �  1    �  �    l  p  L  �  s  =  �  �  �  �  5  �  �  �  '  �  �  �  5  �  �  +  �  H  m  T  �  �  p  W  E  �  �    *  <  m  �  -  �  N  �  �  �     �  5  X  �  �  f  (  q  >  3  �  h<T��%   ;o�D������ě��ě���o��9X�e`B�ě��49X��㼛��8Q�e`B���T���ixռ�`B�]/�]/���0 ż��C��P�`�H�9��{������+��hs������aG��T���u��%�<j�<j�<j���-��t���\)�T����
=�}�P�`��l����w�u���-��+��7L����}�	7L��\)�� Ž��
��/����Q������xս�E������;%�
=q�1'��PBhXA�4�B�B��B&�;B,�BֻB��BL�B�B�B}�B��B!*IB	��B��B^3B#��B	 �B�QBpUB ��BuB��B#iB2}B�1A��JB��BkB{�B((B8�B7�B ��B�B��B��B!7.B��BE`B)ABU�B��B��B�B%�,B'��B�XB�B��B��BUgB�B	CB,J�B�uBO,B+��B.�B
oBڨB0�B�\B`yB��B)4BhB*�B
f*B{DBT�BO�A�m�B?�B}B&>QB+�bB�#B}+BG�B�FB3`BH�B�yB!4�B	��B�B��B#��B	5�B�{B{�B D�BIzB3B#=�B7�B��A�x�B��BBcB��BA�B@BӅB ��B9eB��B�iB ŀBSsB�IB;BDB
��B�wB�cB%�	B(E,B��B�B�B�[B@UB=�B|B+�
B@%B9�B+ͻB.@�B	�&B�B��B@B@B�B��B��B��B
��B �B@�A���A��AxR)A�RfA+�@m��A�#@�jA�@�A�K�Aw��A�`�A�&�A��A|{pA�Y3A���@��FA�Z�Aˤ@���B-KAT�A�8�@^�kAi��A�PA�RA�6UA�j�A���A�yT@&��?�oa@.uA���C�B�@�[@��A��aA�V�A��A��iB`�AДL@�-�A���A���>�y�A[�AY�FAA�@���A!�A�ݝAl�^AO3�A�F�@��A�.A��?��B(A��bA$A2��C�ւA��iA�wA�npAw)C��A��{A�r�Ax�"A�|�A0�@k��Aɐ @�0�A��A�x�Aw!!A�G�A��AĊ�Ay+A�7Aר�@��A�i|A�@�
B��AT��A���@`DAj-{A�zuA�1�A���A�l6A���A�Rh@"?� �@,1#A��IC�X@��3@��A�P�A��A�_'AۀB�)A�}`@�kA�yA���>�M@A\��AY�<A1!@���A �tA��,Al�:AN�AA�q@�%�Al*A�U?��qB<%A��dA�A2��C��CA�}=A�uJA�q�Ay C��               c                               $         %   (                              5   *   !   %   '                           !            9         >      	                  H            &   <         &      )                              K                              -      !      +      '                        )   )         !      #                                 %         #                  #      %            !   #            '                                 ;                              '            #      #                                    !                                       !                           #                     #                              N��NՆ�N3qOO�MP��N�Z�N3$qNҀ�O\�WN��OA��M�x�Nl��NE�P�NFƧO7�!N�5O��fN0sO��OwF�Ny�OT��NPa�N ��OCO��iOi��O?VO��OZ��O���M�V�N[JN��N��#N�f�M
N�N1�O9�"O�O���N4��O���OD�qN�O\ߖOJ"N��N�h�N��N��aO��O�Oa��Nyr�O/S�N��pO�;zO�!mNmO��fO��@O��NϏ�OHJO:^NȜ�O2I�NQо  �  �  ;  �  �  �  �    �  2  V  +  �  �    �  �  �  �  �  �  �  H  �  N  5  �  X  �    �  �  '    �  }  �  �    c  A  �  �  I    �    c  	C  A  e  h  �  ]    ]  
�  e  �  �  P  �  	�  �  �  -  H  �  �  	�  �  �<�o;�`B;ě�;o��1%   ��o�D����o��o�o��`B��1�#�
�e`B�49X����o��󶼬1���ͽo���ͼ�/��h���\)�+�u�@����,1�\)�t��<j����w�8Q�,1�0 Ž0 ŽH�9�@��8Q�8Q�q���H�9�H�9��t��e`B�T���u�Y��T���aG��e`B��-��+��7L��7L���㽑hs���w���-���-������ Ž����"ѽ�/��`B����������������������:;CHLT_akdaaTLHE=;::��������������������7<HUanusnla_ULH<7177
0Ub����}eU<0%��
����������������������������������������������������������������������������������������������������"16BOP[ad^[OB6)��


������������"#&/47/-#""""""""��������������������;ER[gt��������g[B97;Y[ahtzztthf[YYYYYYYY��
#3<></#�������	

��������25BN[gt�����gYTNB522����������������������������������������lmz������������zskil������������������������������������������������������������������������������������������1/03:HTadca]TMH;/,,1��������������������()26BO[agfa[OHB63.)(FKOX[ht}����uh[OB>>F������������������������������������eht���xthgeeeeeeeeee��������������������./<HOUUUSNHA<40/....�������

���������@BOOO[[^^[XOBB=<@@@@����������������������
 
������������"#/7<C=<9/$#""""""""�������������������������� ��������r{���������������~sr&)688A6)'&&&&&&&&&&&�������������'-8<IMMMOJC<0#SU`bn{|{rnbWUQSSSSSS(06<O[htztsrlh[OB6/(���������?BN[gitvvtg[NB57????))//)"	)679976)$X^dhmt�����wtih[WVX58:BNRgt���tg[NB6015��������������������5<HUdlnpnjaUHE<:5205�������������������������������������������
 ��������Zgt�����������g[ROPZkpz������������znihk����������������������������������������������������������������#/3<?BC<<:/#}����������������u}})35BENXNJB55)"qt���������thiqqqqqq���������}zbWRUaknw�nronca_UUUV[annnnnnn��������������������������������������������������
��#�$�&�#���
������Ŀ������������ĿͿѿѿѿĿĿĿĿĿĿĿ����������������������������������������˽����z�������нݽ���(�>�Q�I�4����н����������!�-�:�?�?�:�4�-�!�����ÇÆ�~ÆÇÓØàâàÓÇÇÇÇÇÇÇÇÇ������ܻлλлջܻ�����������������������������������
����
�������¦­²¶¸²¦���������������������ĿѿԿؿ׿ѿ˿ȿĿ��z�x�y�zÇÓÖÓÑÇ�z�z�z�z�z�z�z�z�z�z�<�;�<�?�H�U�_�a�i�a�U�H�<�<�<�<�<�<�<�<�H�B�=�=�H�U�^�Z�W�U�H�H�H�H�H�H�H�H�H�H�ݿѿ������������Ŀݿ��������������/�,�#���#�/�<�C�>�<�2�/�/�/�/�/�/�/�/�������!�)�2�6�D�N�T�]�O�B�6�*�������������������������ɼʼ��������������������}�o�i�h�m�}����������������������àÙÖÛàéìôðìàààààààààà�l�_�S�F�3�,��!�-�:�S�_�l�x���������x�l�������������������������!�������׾ҾԾ׾ؾ��������׾׾׾׾׾׾׾��H�@�;�3�.�&� �"�/�;�H�T�_�a�e�g�f�a�T�H������������!�"�!�����������������m�e�`�T�L�T�`�m�y�z�y�p�m�m�m�m�m�m�m�m�n�f�b�V�U�I�H�I�R�U�b�n�{ŀŇŇ�{�v�n�n�g�Z�N�A�5�*�(�-�5�A�O�Z�}�������������g�T�N�H�F�G�L�R�T�a�m�z����������z�m�a�T���x�s�s�u������������������������������"��	�������"�/�;�H�X�a�b�_�\�V�H�;�"������ĿĸĳĳļĿ���������
���
������~�r�|�����������ɺֺݺ����ʺ������~�r�p�q�q�r�t�~����~�r�r�r�r�r�r�r�r�r�r�����������ɺֺ׺ֺԺɺ���������������������������������������������������������D�D�D�D�D�D�D�D�D�D�D�EEEEEE D�D�D컪���������������ûȻл׻ллû����������������������������������������������������������������������������������������������������������������������������������[�O�B�@�A�B�K�O�[�h�i�tāĂĄĂ�w�t�h�[�[�P�O�K�K�O�[�f�h�t�xāčĔčČā�t�h�[�b�V�I�?�=�I�O�W�[�^�b�o�{ǀǉǌǈ�{�o�b���������������������������������������Ҽ4�'�!��"�'�4�@�M�Y�r�����������Y�M�4�������������������������������������������|�s�s�k�q�s�v�������������������������ܹϹɹù������ùϹ׹ܹ�������������ܿ	������������	��!�.�2�;�8�.�"��	������������	������	�������ʼǼ��ʼ˼ּ���������ּʼʼʼʼʼʼr�h�o�r���������������r�r�r�r�r�r�r�r�����������z�����������������ĽŽȽĽ����
�����������'�C�Q�[�Y�M�I�H�C�6�*��
�m�h�`�b�e�h�m�y�������������������y�m�m���������������ʾо׾����پ׾ʾ�����Ň�{ŇŌŔŠŭűŹźŹŭŠŔŇŇŇŇŇŇ�r�i�l�r�u������������ɼ�������������r��ڼ���������������������ĿĶıįįįĳľ����������������������Ŀ��ܹԹȹǹܹ���3�@�L�V�]�[�S�B�'��������������������������������������²�s�q�u¦��������������¿²�G�=�-�(�/�:�G�S�`���������������l�`�S�G������߽޽�����(�0�+�4�9�(�!���FFFFF!F$F1F=FJFJFJFIFAF=F1F$FFFF���
�����������
��#�,�/�8�:�/�'�#��àÙ×àâëìóùÿ������������üùìààÞÙÕÙàìõùüýýùìàààààà���������ĿѿٿٿտѿϿĿ���������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� 8 = U S ( M D ? . 7 A 6 & R 0 ? _ $ ^ W V % q : ` X   l . B ( P a � @ + H + � x a  / V U ? C � 8 ? b * I T j Q  [ Q O < P ; P 9 Z 1 ( G O � /  (  �  Q  S  s  �  \    �  �  �    w  |  �  l  �  �  �  s  �  �  �  �  �  5  *  �  �  �  9  �  �  5  t  �  +  �  H  m  T  �  X  p  W  �  �  �  �  �  <  �  �  -  �  N  �  �  �  �  A  5  4  �  �  p  �  6  >  �  �  h  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  A7  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  	  �  �  �  �  �  �  �  �  t  _  J  6      �  �  �  �  X  %  ;  6  1  ,  '      �  �  �  �  �  }  e  M  C  =  8  2  -  T  h  x  �  z  l  a  W  P  P  U  [  K  )  �  �  l    �  d  f  �  Y  �  �  �  �  �  �  $  �  w    �  %  l  �    o   �  �  �  �  �  �  �  �  �  �  �    w  �  �  �  �  �  }  n  _  �  �  �  �  �  �  �  �  |  c  J  2    �  �  �  �  m  =         �  �  �  �  �  �  j  E    �  �  �  }  N    �  �  �  �  �  �  {  g  L  +    �  �  �  Y  +  �  �  �  f  )  �  7  2  )  !         �  �  �  �  �  g  ?    �  �  Z    �  �  7  C  L  R  V  U  Q  L  C  5     
  �  �  �  �  Q    �  	  +  8  E  R  _  i  m  r  v  z  {  {  z  z  y  n  `  R  E  7  �  �    /  Y  �  �  �  �  �  �  �  �  �  w  .  �  �  ;  �  �  �  �  �  �  �  �  �  �  �  }  r  m  ^  M  6      �  �  �  �       �  �  �  �  �  a  7    �  �  o  +  �  X  �  C  �  �  �  �  �  �  �  �  �  �  �  �  }  y  s  m  g  a  [  U  �  �  v  o  ~  �  |  o  Z  9    �  2    �  �  e    �  <  �    f  �  �  �  �  �  �  �  �  �  �  c  9  
  �  q     }  8  c  �  �  �  �  �  �  �  �  M    �  �  d  /  �  O  �  J  �  �  �  �  �  u  h  [  M  <  *      �  �  �  �  �  �  �  u  �  �  �  w  \  ?    �  �  �  `    �  �  H  �  ~  �  |  �  �  �  �  �  �  �  �  �  �  �  �  n  A    �  _  �  y  .  H  =  2  &      �  �  �  �  �  �  �  �  �  �  �  t  b  P  �  �  �  �  �  �  �  �  �  �  �  q  S  .     �  �  O  !  	  N  N  N  M  M  M  M  M  L  L  E  7  (      �  �  �  �  �  5  .  '         
  �  �  �  �  �  �  �  �  �  z  h  V  D  �  �  �  �  �  �  �  �  �  �  m  J     �  �  �  /  �  ^  y  X  D  /      �  �  �  �  ~  a  >    �  }  0  �  �  )   �  �  �  �  �  �  �  �  �  �  �  �  �  �  K  �  |    |  �  �  �  K  �  �  �  �       �  �  �  �  k  ,  �  d  �  ?  �  �  �  �  �  �  �  �  �  �  i  D    �  �  �  d  4  �  �  1  @  a  �  �  �  �  �  �  �  m  M  )    �  �  :  �  L  �  �  �  '    �  �  �  �  s  W  @  $  �  �  �  0  �  K  2  	  �  �        &  -  5  =  E  M  U  Y  Y  Y  Y  Y  Z  Z  Z  Z  Z  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  �  �  I  r  q  ]  E  .    �  �  �  �  v  O  #  �  �  �  ?  �    �  �  ]  ,  �  �  �  s  >  �  �  �  �  �  �  ]  B  6      �  �  �  �  �  �  y  a  D     �  �  t    �    �  �  `   �    #  (  ,  1  4  4  3  3  3  3  3  3  3  3  3  2  2  2  1  c  ]  W  P  J  D  >  8  3  .  )  $            �  �  �  A  :  2  +  #            �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  B    �  �  J  �  �  �  �  �  �  �  �  }  Y  /  �  �  p    �  )  �  9  �    I  *    �  �  �  �  �  �  �  x  O  (  �  �  �  X    �  �      
    �  �  �  �  �  �  �  �  �  �  c  �  [  �  �  h  �  �  �  �  �  �  �  �  �  `  !  �  n    �  X  �  ?  W  �      �  �  �  �  �  �  �  �  j  F    �  �  �  �  �  v  R  c  ]  X  S  M  H  C  >  8  3  ,  #         �   �   �   �   �  �  �  �  �  �  	8  	B  	:  	  �  �  ,  �  r  �  ;  W  b  [  M  �  -  :  @  9  0  '      �  �  �  _    �  �  ,  ~  �   �  e  e  e  _  W  L  9  &    �  �  �  �  �  t  \  F  6  1  +    6  R  `  e  h  h  c  [  O  >  (    �  �  %  �    e  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ]  V  O  I  B  7  %    �  �  �  �  \  0     �  M  �  =   �      �  �  �  �  �  �  �  �  �  �  �  ]  6    �  �  H  �  ]  R  H  =  0  "      �  �  �  �  �  �  }  d  J  1     �  	J  	�  	�  	�  
  
k  
�  
�  
c  
'  	�  	w  	  ~  �        X  �  e  _  Y  R  L  E  >  6  .  '          �  �  �  �  �  �  �  �  �  �  m  S  :      �  �  �  �  S  !  �  �  ~  :  �  �  �  �  �  �  y  V  0  	  �  �  �  U  "  �  �  �  `  9  :  �  I  N  N  C  6  !  �  �  �  i  1  �  y  �  �  �  [  �  �  �  ~  z  n  K  %  �  �  �  �  m  I    �  �  J  �  !  '  �  	;  	U  	�  	�  	�  
A  
{  
�    �  �  �  X  �  �  4  �  w    �  �  �  �  �  W  -    �  �  �  f  F    �  �  O    �  3  �  �  �  �  �  �  u  R  ,    �  �  m  5  �  �  V  �  �  �  �        +  ,  '       �  �  �  }  Y  A  C    �  �  X      5  C  H  D  7  $    
�  
�  
=  	�  	�  	<  �  y    �  :  �  n  �  �  u  d  Q  @  6  0  )  "      	      �  �  �  ^  �  �  �  �  h  C    �  �  �  M    �  X  �  �  L    �  H  	`  	�  	�  	�  	u  	h  	X  	3  �  �  f    �  Q  �  |    �  K  
  �  �  `  7    �  �  �  �  s  U  "  �  �  f  +  �  �  I  �  �  m  �  �  ?  �  �  @  �  �  m    �  d    �  _    �  �