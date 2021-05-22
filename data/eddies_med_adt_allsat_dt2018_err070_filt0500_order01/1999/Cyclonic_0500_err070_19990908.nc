CDF       
      obs    G   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�9XbM�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�x�   max       P��       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��l�   max       <��       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�Q��   max       @F�            !    effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @vu��R       ,   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @Q            �  70   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��           7�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��+   max       ;o       8�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B4h6       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B4V�       ;   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =��/   max       C�{&       <0   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >=��   max       C��        =L   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          O       >h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          K       ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          I       @�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�x�   max       P���       A�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�bM���   max       ?٭B����       B�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��l�   max       <�9X       C�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�Q��   max       @F�            E   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(�    max       @vup��
>       P(   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @Q            �  [@   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��           [�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�       \�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?w�����   max       ?٠'RT`�     0  ^                              !      	            1      
   6            	      	      O                                  ?         
               0      
   (         	         @            K         
            
      	         O7PfN�`M�x�O5��O��N��NG�LOT>O��P��NZM�N�v�N�@O���OfB}O�d�M��O)��Pw5N�rzO��OGL�N�I�M�B�N�+�O/l�P%�N�;qO<̀N�#fO�M:N:O+��Oa�pNC�KP'�\OS�BP��NY�9M���O)�yN6yN#S�N[7�OR�RPVpNJw2N��O��N4G�O/z�Nٯ�N�,N�[�O�1N��N�HO��O��N�t�O7�N��OBN��ON�r�N~8)NLJ�N��N�կO��<��<t���o�o��o��o�ě��#�
�49X�D���e`B�u��o��o��C����㼛�㼣�
���
���
���
��1��9X��j������/�o�C��\)�t��t��t���P��P��w��w��w�#�
�#�
�,1�,1�49X�49X�8Q�@��@��T���]/�aG��ixսixսm�h�m�h�m�h�u��+��7L��7L������P�������-���-�� Ž�-��Q콺^5��^5��j���`��l�)47=@B>63)KO[[]hipnhhb[OMFDDKK���������������������������������������������
#'#
������zz������������zqz�zz��� �������������
#06@D?<90#

������ �������������"'0ENSUamqr|��zaH/%"Z[`fghjtuutrkh[XZZZZ�������������@HQUYXUHD>@@@@@@@@@@��������������������@BHN[gtz�����tg[NF?@������	�������������������������������HKacnz�����zwna[UQHH+6BO[h~��{[O6����������������������������������������������������������nnaUPHAGHUadnqnnnnnn��������������������&).6BEOUTOBB?6)&"$&&
/<HU\WULH?</#x��������������zurxx����


���������qt~���������������tq�������������������� )AN[gt����tgYN5/)$ ��	
!!
�����������������	��������6=COX^hmqtph\C<60/26���������������������������������������������������������(OYS6�������\hhrt�����th\\\\\\\\)26BHB6)(LO[hjt���������tg[OLahu����uh`aaaaaaaaaa��������������������:<?IUVUQOID<87::::::"/6;HILNNLH;/("������!������������������������������������������������GHNUWanpoonicaUMHHGG������������������������������������������������������������45BEN[cd`[NB53444444gmqt~������������tgg9>HUanz����qaUH>9769NOUZ[\chih^[VONNNNNN��
#$/8/#
������?HMUnrs{~znaUPHAA?�����

������:;;HTaeeaaXTHE<;::::"$)/168<BHSTTRNH</#"##-02<<??=<;0.,(####.0<@DIILMJI<0-&%&(..rtx{����������|vtrrr_abnryz~���|znjb_]^_��������������������Y[bgpqqmg[USYYYYYYYYrtx�������trrrrrrrrrMNQX[gmmjg][NNNLMMMM��	 ������P[agt���������tng^[P�ɺ��������������ɺֺ�������������ɿ��������������Ŀѿݿ߿�����ݿѿĿ����T�R�T�Y�`�j�m�o�n�m�`�`�T�T�T�T�T�T�T�T�����������������������������;̾ξɾɾ��~�t�q�o�p�r�t�y ¥���������������������������������������������	����	��������������������лû��������������ûлӻܻ����ܻһ����s�g�Y�Z�s����������������������������������ĳĦĚčāăčĦ�����
��������.�+�"��	��	���"�%�.�;�?�=�;�.�.�.�.���������)�5�9�;�5�3�)����������������������������������������꿫�����z���������Ŀѿݿ������ݿѿ����m�g�m�o�h�i�j�m�s�z�����������������z�m�-�$�)�5�=�A�G�T�m�����������y�m�`�G�;�-�U�T�U�]�a�c�n�o�q�n�n�a�U�U�U�U�U�U�U�U�R�H�=�;�6�7�;�<�H�T�Y�a�m�v�s�m�e�a�U�R�����x�c�S�I�F�F�N�d�x�~�����������������x�x�p�l�c�_�\�]�_�l�m�x�y���������z�x�x�L�H�@�3�3�L�Y�e�r�~����������~�r�e�Y�L�Y�M�E�>�>�@�O�Y�f�r�u���������r�n�f�YààââàÝÓÇÅÀÂÇÒÓàààààà�����������������������������������������������������������������"�%�(�(�)�6�7�B�F�O�Q�M�K�B�6�)�������'�3�@�L�e�~�����������~�Y�'��s�k�g�c�g�l�s�������������������s�s�s�s�����Լʼȼ������ʼּ��������������6�5�4�6�9�B�O�Z�Z�Y�O�B�6�6�6�6�6�6�6�6���׾ھ������	�������	����������������������������������������������@�4�@�A�J�T�Y�`�f�r��������w�r�f�Y�M�@���׾˾׾����	��"�(�)�$�"���	���ﾌ�������������������������������������������l�^�X�_�y�����ܻ������ػлû�àÜÓÌÅÃÇÐÓàìù��������ùòìà�2�3�-�����G�y�����Ľ����н����G�2����������������������������������������ù÷óùü������������úùùùùùùùù��������¿¶¶¿�����������������������ؾ��������������������������������������������������������������������������������������������������������������������������������~�������������������������������M�M�f�~�������ʽ�������㼱���v�f�M�������������������������������������������������$�.�$�����������������ù��������������ùϹܹ��������ݹܹϹ���ż�������������������������������������������������*�6�C�C�<�6�2�*�����ŠŜŘřŠŭųŹ������������ŹŭŠŠŠŠ�I�H�>�=�:�;�=�I�V�Z�a�b�X�V�I�I�I�I�I�I�H�<�0�*�0�3�<�I�L�U�X�b�e�b�Z�U�O�S�J�H�׾ʾ����������ʾ���	�"�/�1�+�"��	��׹ù��������������ùŹϹѹϹǹùùùùùÿݿۿѿɿǿпѿۿݿ�������ݿݿݿݿݿٿѿͿͿѿݿ�����#�0�3�$�������ED�D�D�D�D�D�D�D�EEE*E7E?E>E5E&EEE��������������(�+�+�(������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�Dӽ������������������ĽͽнڽнĽ������������޽��� ����(�*�4�,�(���������w¦²¸¿��¿²¬¦E�E�E�E�E�E�E�E�E�E�E�FFFFE�E�E�E�E�����������������������������������������n�j�a�Y�a�n�zÇÊËÇ�z�n�n�n�n�n�n�n�n�<�6�/�)�/�4�<�H�I�N�Q�H�<�<�<�<�<�<�<�<�s�o�g�Z�X�U�Z�g�s�w�������������s�s�s�s�����������������������������������čĄāĀ�āĈčĔĚĦīĬıĪĦĢĚĖč & ( � g = ^ n , 1 O h N 6 C 9 3 Z @ , w N = T 4 2 k '  | 1 f n Z . @ C ) R n ~ C q ; c * � Y p M ? * M D � * J S = 4  K a G Z T 2 P . g n \    �    b  �  i    �  �    �  �  �  ?  �  �    C  �  �  �  }  �  �        �    1  �  w  p  �  �  ]    �  �  �  �  �  �  '  n  �  �  {  �  g  H  w    �  �    =  �  !    �  �  �  0  �  d  �  r  S  �  %  z;o��o��o�o��9X��o�t����#�
�8Q켃o�ě�����\)��㽋C��ě������P���+�+�������C��P�`��G��8Q�P�`�@��aG����]/�]/�49X��hs��C����`�49X�<j�T���D���H�9�D����\)����u��o�\�u��\)��7L��C���%���m���㽝�-���T��+��9X��
=��-�ě���vɽ�l������������ȴ9���+B� B#�B!ʡB ��Bz�BIUB)�B%BB�A�$�B��B$�BW�B+_pB	9+B��B"0"B�Bn�B �dB"�VB"mB��B��B�B�B�BF<B  KB��B�SB$UGB�{B1TB YB�B
�BǲB�B{�B+B2�DB4h6B&��A���B-LB9XB��B%B��B�{B%�B�B
L�B�B>B9�B4�B*�A�rB�HB%��B&�B
RGB��Bj�B	.�B
?QB�XB?-B	��BƚB3�B!�B!��B��Bb�BMBB%@�B,�A���B��B�^BGsB+A�B	6�B>�B"pBKB��B �RB"C4B"��B�hB>ZB�.B��B�iB,�B �,B��B�B$s�B��B1�BS�B��B��B�$BbB8nB>dB2�B4V�B&��A���B-?jB>�B��B��B��B��B<�B+4B	�PB��B@�B�=B@~B�"A��YBŊB%��B&9�B
?�B�B�EB	,�B
A�B	(OB;�B
<O@>GAy��Ah@LAN3�A��A�u�A�t@�RA�J�A�$�A_DbA�9A���Ax>A�ǃAh=�AƉ�A�Kz@�oc@�7C?�5�@ݒ�A�j�?tiA���A��D?�~A���A�A��AXܞA�~�@��AZ	�AI!�@�	A���AA#�]A��,A�k�AKa�AJ��A���A��@��:A��B�?>���A�b�A�QyA�+�Bf�A���AV�y=��/A|��A��C�a.A���C���A$J�A23�A���C�{&A�p�A�%oA���A��AѣlA�+�@C2ZAyI:AiyAK�A�m�A���A�o�@���A�ZA�j�A`�A���A�	�Ax�A��[Ai 
AƕZA��@��@���?���@�yCA�|g?pOA�7�A�}?��A���A3A�kwAYxA�)�@��AZȼAH��@���A�{OA ėA#�GA͏A�N}AJ��AJ�iA���A�VA ��A���B��>�^"A�}�A�z A��BJaA��AX�b>=��A}2�A��$C�W�A��jC���A"�A3�FA�y�C�� A�:A�bA�~5A�r'A҉A�w�                              "      
            2         7            
      	      O                           !      ?                        0   	      )         
         @            K                     
      	                                    #   -            #      #         )                        '                           -      K                        7                           #                                                                              -            #               #                        !                           -      I                        )                                                                           N�=�Ni�M�x�O5��O��NT��NG�LO!�N��1P��NZM�N�v�N�@O�O1�O���M��O)��P $�NE��O��OGL�NX)�M�B�N�+�O�zO�]N��O<̀N�#fO���N:O+��Oa�pNC�KP'�\OS�BP���NY�9M���O)�yN6yN#S�N[7�OR�RO�ҒNJw2N{�O��N4G�O/z�Nٯ�N�,N�[�O�bXN��N�;�O��Oz��N�t�O7�N��OBN��N���N�r�N~8)NLJ�Nfe3N�կO��  �  g  b  �  �  �  ,  �  �  `  @  m  Y  �  �  �  /  �    �  u  s  F  ?  �  �  	2  �      �  �  �  �  e  �  z  `  �  �  �  K  �  a  �  �  N  �  
�  4  �  �  .  E  �  !  �  �  �  �  �      �    �  �  �  �  �  0<�9X;��
��o�o��o��`B�ě��T����/�D���e`B�u��o��C���9X��󶼛�㼣�
�ě���9X���
��1��j��j������h�]/�t��\)�t���P�t���P��P��w��w��w�'#�
�,1�,1�49X�49X�8Q�@��]/�T���aG��e`B�ixսixսm�h�m�h�m�h��\)��+��C���7L���㽗�P�������-���-�� Ž�Q콸Q콺^5��^5��vɽ��`��l�)56764,)IOR[`hjhf^[OOJIIIIII���������������������������������������������
#'#
����������������������������� �������������!#)07<=<:50#��������������������"'0ENSUamqr|��zaH/%"Z[`fghjtuutrkh[XZZZZ�������������@HQUYXUHD>@@@@@@@@@@��������������������LN[cgkt}����wtg[SNKL��������������������������������������HKacnz�����zwna[UQHH.6BO\hx~~x[OB6����������������������������������������������������������CHJUacnnngaYUHCCCCCC��������������������&).6BEOUTOBB?6)&"$&&
#/<GRJH=</#
zz~��������������{z���� 

���������qt~���������������tq��������������������")5BN[gt���tg[N51)"��	
!!
�����������������	��������6=COX^hmqtph\C<60/26���������������������������������������������������������'NP6�������\hhrt�����th\\\\\\\\)26BHB6)(LO[hjt���������tg[OLahu����uh`aaaaaaaaaa��������������������:<?IUVUQOID<87::::::"/6;HILNNLH;/("�����������������������������������������������������HHNUXanooonmibaUMHGH������������������������������������������������������������45BEN[cd`[NB53444444gmqt~������������tgg=BHUanx|�{tnaUH<:9=NOUZ[\chih^[VONNNNNN
"#/0/#
��?HMUnrs{~znaUPHAA?����

���������:;;HTaeeaaXTHE<;::::"$)/168<BHSTTRNH</#"##-02<<??=<;0.,(####.0<@DIILMJI<0-&%&(..rtx{����������|vtrrraannuz}���zynkdaa__a��������������������Y[bgpqqmg[USYYYYYYYYrtx�������trrrrrrrrrMNORY[gklig[[NMMMMMM��	 ������P[agt���������tng^[P�ֺϺɺ����ɺκֺٺ��������ֺֺֺֿĿ����������Ŀ˿ѿݿ�޿ݿѿĿĿĿĿĿĿT�R�T�Y�`�j�m�o�n�m�`�`�T�T�T�T�T�T�T�T�����������������������������;̾ξɾɾ��~�t�q�o�p�r�t�y ¥���������������������������������������	����	�������������������仧�������������ûĻлܻ����ܻлû������������}�s�r�s�����������������������������ĳĦĚčāăčĦ�����
��������.�+�"��	��	���"�%�.�;�?�=�;�.�.�.�.���������)�5�9�;�5�3�)����������������������������������������꿫�������������Ŀѿݿ������ѿĿ����z�y�p�m�m�m�p�z�����������������������z�;�4�1�6�<�C�G�T�`�m�y�������y�m�`�T�G�;�U�T�U�]�a�c�n�o�q�n�n�a�U�U�U�U�U�U�U�U�R�H�=�;�6�7�;�<�H�T�Y�a�m�v�s�m�e�a�U�R�h�S�L�I�I�S�h�x���������������������x�h�x�t�l�e�_�_�_�l�u�x���������x�x�x�x�x�x�L�H�@�3�3�L�Y�e�r�~����������~�r�e�Y�L�Y�M�E�>�>�@�O�Y�f�r�u���������r�n�f�YÓÊÇÂÃÇÓÓÔàáàßÚÓÓÓÓÓÓ������������������������������������������������������������������#�&�)�6�B�D�O�P�L�J�B�?�6�-�)��e�Y�@�3�(��"�+�@�L�Y�e�~���������~�r�e�s�o�g�e�g�o�s�������������������s�s�s�s�����Լʼȼ������ʼּ��������������6�5�4�6�9�B�O�Z�Z�Y�O�B�6�6�6�6�6�6�6�6����ؾ۾������	�������	��������������������������������������������@�4�@�A�J�T�Y�`�f�r��������w�r�f�Y�M�@���׾˾׾����	��"�(�)�$�"���	���ﾌ�������������������������������������������l�^�X�_�y�����ܻ������ػлû�àÜÓÌÅÃÇÐÓàìù��������ùòìà�6�4�.�� ��G�y���Ľ�����н����G�6����������������������������������������ù÷óùü������������úùùùùùùùù��������¿¶¶¿�����������������������ؾ��������������������������������������������������������������������������������������������������������������������������������~����������������������������������������ļּ����������㼱�����������������������������������������������������$�-�$�����������������ù��������������ùϹعܹ��������ܹϹ���ż�������������������������������������������������*�6�C�C�<�6�2�*�����ŠŜŘřŠŭųŹ������������ŹŭŠŠŠŠ�I�H�>�=�:�;�=�I�V�Z�a�b�X�V�I�I�I�I�I�I�H�<�0�*�0�3�<�I�L�U�X�b�e�b�Z�U�O�S�J�H�׾ʾ��������ʾ׾���	���(�+�%��	��׹ù��������������ùŹϹѹϹǹùùùùùÿѿʿȿѿӿݿݿ޿�����ݿѿѿѿѿѿѿݿٿѿͿͿѿݿ�����#�0�3�$�������D�D�D�D�D�D�EEE*E7E>E=E7E3E%EEED�D���������������(�+�+�(������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�Dӽ������������������ĽͽнڽнĽ������������޽��� ����(�*�4�,�(���������w¦²¸¿��¿²¬¦E�E�E�E�E�E�E�E�E�E�FFFFE�E�E�E�E�E�����������������������������������������n�j�a�Y�a�n�zÇÊËÇ�z�n�n�n�n�n�n�n�n�<�6�/�)�/�4�<�H�I�N�Q�H�<�<�<�<�<�<�<�<���v�s�g�Z�Y�V�Z�g�s�u�����������������������������������������������������čĄāĀ�āĈčĔĚĦīĬıĪĦĢĚĖč " ! � g = j n ) > O h N 6 G   - Z @ & r N = Q 4 2 e   | 1 c n Z . @ C ) Q n ~ C q ; c * f Y v J ? * M D � * J V = 4  K a G Z K 2 P . ^ n \    �  n  b  �  i  �  �  X    �  �  �  ?  �  F  G  C  �  8  �  }  �  �      �  �  �  1  �  Y  p  �  �  ]    �  �  �  �  �  �  '  n  �  g  {  �  Q  H  w    �  �  �  =  �  !    �  �  �  0  �  %  �  r  S  �  %  z  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  4  T  s  �  �  �  �  �  �  }  b  A    �  �  �  ]  )  +  �  K  O  T  Z  `  d  g  g  e  _  V  H  8  &      �  �  �  �  b  `  _  ^  ]  \  Z  S  I  ?  4  *       
   �   �   �   �   �  �  �  �  �  �  �  �  �  �  u  j  a  W  L  >  1     �   �   X  �  �  �  �  �  t  W  3    �  �  k  2  �  �  f  &      W  0  \  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  h  e  h  ,  #          �  �  �  �  �  �  �  �  �  �  �  y  k  ]  �  �  �  �  �  �  �  �  �  �  �  �  d  <  
  �  v    �  l  �      7  S  l  ~  �  �  �  �  �  �  �  b  ;    �  �  K  `  O  6    �  �  �  �  c  E  +  
  �  �  v  5  �  �  E  i  @  =  ;  8  5  2  0  -  *  '  "            �   �   �   �   �  m  _  P  A  2  $      �  �  �  �  �  �  �  r  `  Z  �    Y  T  P  K  F  B  =  8  4  0  +  '  "    
  �  �  �  �  �  �  �  �  �  �  �  |  r  j  d  X  E  /    �  �  �  u     �  O  o  �  �  �  �  �    g  N  .    �  �  �  X  $  �  �  !  H  |  �  �  �  �  �  �  �  e  2  �  �  H  �  W  �  +  z  F  /  '         	    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  U  9  $      �  �  �  �  �  �  �  �  �  �      	  �  �  �  �  x  ^  F  /    �  ~  +  �  �    %  s  �  �  �  �  �  �  �  �  {  v  |  �  �  �  �  x  C  	  �  u  g  Z  O  Y  S  D  5  %      �  �  �  �  �      3  b  s  e  W  I  8  )  "    �  �  �  �  ~  f  R  ;  #  	  �  >  %  5  E  @  :  3  -  $    �  �  p  L  '     �  �  �  X  ,  ?  5  +  !        �  �  �  �  �  {  d  Q  >  +       �  �  �  �  �  }  q  a  Q  A  1  "      �  �  �  �  �  S    J  q  �    x  v  ^  D  #     �  �  {  M    �  T  �    O  �  �  	  	  	"  	0  	-  	  �  �  y  $  �  3  �  �  U  �  �  �  �  �  �  �  �  �  �  �  }  d  J  2      �  �  �  �  �        �  �  �  �  �  �  �  �  �  z  K    �  �  p  C  $         �  �  �  �  �    c  F  (    �  �  �  G  �  �  b    �  �  �  �  �  �  y  \  A  *    �  �  �  v  N  /    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  j  _  �  �  �  �  �  �  �  �    Y  .  �  �  �  U    �  �  ~  �  �  �  �    p  ^  I  1    �  �  �  �  s  N  &  �  �  �  5  e  Z  P  E  :  /  #      �  �  �  �  �  �  �  d  F  '  	  �  ^  )  �  �  ~  9  &    �  �  �  �  �  �  w  ]  ,  �  f  z  p  a  L  2    �  �  �  Y    �  �  E  �  �  
  i  �   :  N  I  0  $  �  �  y  @    �  �  �  �  �  f    �  -  �  �  �  �  �  �  �  �    t  j  _  S  G  :  .  !    �  �  �  �  �  �  }  k  Z  I  9  )    	        
      �  �  �  �  �  �  �  �  �  }  r  j  _  S  @  (    �  �  �  m  =     �  K  E  ?  9  2  ,  $          �  �  �  �  �  �  �  }  h  �  �  �  |  q  f  [  O  B  3  #      �  �  �  �  �  �  �  a  [  V  P  K  E  @  9  1  *  #         �   �   �   �   �   s  �  �  �  �  �  �  z  `  B    �  �  �  D    �  j    �  y  {  �  �  �  �  �  �  X    �  �  g    �  y  "  �  *  ?   �  N  S  X  [  _  \  K  9    �  �  �  n  =    �  �  o  7     �  �  �  �  �  �  r  [  H  ;  *    �  �  �  q  >    �  �  
�  
�  
�  
u  
?  
  	�  	  	5  �  �  J  �  �  8  �  c  �    T  4  -  &        	    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  j  U  :    �  �  �  �  �  r  [  o  �  �  �  �  �  �  �  }  h  W  G  :  /      �  �  �  �  a  �   �  .  (  "    	  �  �  �  �  �  �  e  H  ,    �  �  �  8  �  E  `  |  �  �  �  �  �  y  t  n  i  c  \  V  O  F  >  5  ,  s  �  �  �  y  k  O  "  �  �  |  =  �  �  ,  �    t      !  A  \  H  1    �  �  �  |  I    �  �  h  .  �  �  z  =  �  �  �  �    h  Q  9  "    �  �  �  �  �  �  o  Q  F  <  �  �  �  �  �  �  �  k  P  3    �  �  �  �  t  B  �  �  /  z  �  {  P    �  �  W  �  {  �  P  �    
T  	u  T    n  ^  �  �  �  g  C    �  �  �  _  +  �  �  �  I    �  O  �  x  �  �  q  8  $    �  �  �  �  x  D    �  a  �  `  w  �  �    
  �  �  �  �  �  s  W  3    �  �  g  '  �  �  >   �   �    x  o  ^  G  *  
  �  �  �  c  3    �  �  [    �  v    �  �  �  �  �  �  e  B       
  �  �  �  �  �  e  8     �  �        �  �  �  �  �  ]  3    �  �  W    �  �  0  �  �  �  �  �  u  `  I  /    �  �  �  �  �  `  @  $  
  �  �  �  �  �  w  ^  D  (    �  �  �  �  a  7    �  �  �  �  �  �  z  l  [  H  3    �  �  �  y  ;  �  �  ~  =  �  �  t  0  i  t    �  �  �    t  e  W  G  5  #    �  �  �    �  %  �  �  �  �  ~  X  0  �  �  x  /  �  �  k  ?    �  �  �  b  0    �  �  �  �  n  <    �  �  r  G  (    �  �  F  �  ,