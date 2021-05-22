CDF       
      obs    V   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?���vȴ:     X  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P�/     X     effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �� �   max       <e`B     X   \   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?!G�z�   max       @F��Q�     p  !�   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��G�z�    max       @vq\(�     p  /$   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @Q            �  <�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�9        max       @��`         X  =@   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �O�   max       ;�o     X  >�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��\   max       B4��     X  ?�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�   max       B4��     X  AH   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       <1�   max       C�[C     X  B�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�0�   max       C��     X  C�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          i     X  EP   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?     X  F�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5     X  H    
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       Py�(     X  IX   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���+j��   max       ?ڈ�p:�     X  J�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �� �   max       <e`B     X  L   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @F��Q�     p  M`   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �θQ�     max       @vq�����     p  Z�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @Q            �  h@   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�9        max       @�          X  h�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @^   max         @^     X  jD   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�R�<64   max       ?ڈ�p:�     �  k�   
         (               !      =   6            h      	                        9      )         
                     	               2                           #                                                            O   '   !         
         )   5         3      N�ߟOU	IN8B�O��@N+V<O�P�N��FN5��P�OG��P�/P6��O��O|$O�L<P.�.NН�N�?�N��EON=N��O��ORm�N�p�N͘LP�Y	O�XO�NUO��HN#O��N��NɎ�N�oN��N�J�OR��N5�Nƽ�OAtNe'AO��P6��N:�N���N��=O�O�O�OU.O��!Of^�O�B�M��O��+OU?O��N}e�O���Oac�O��N]rJO�sN/~NA�9O�)O�\N�4O:��O�N�j�N�)�Oa3�O�.�O[�kN���NlNe� N��O�5OK1O�rN�]�N�mO�jfOaU�N?$�<e`B<D��;�`B;��
�ě���`B�t��t��#�
�#�
�49X�D���D���D���D���D���T���T���u��o��o��t����㼛�㼛�㼛�㼛�㼣�
���
���
���
���
��1��1��9X�ě��ě��ě��ě��ě��ě����ͼ�����/��`B��`B��`B��h��h��h�t��t��t��t���P��P�#�
�#�
�#�
�#�
�'0 Ž0 Ž8Q�8Q�8Q�L�ͽL�ͽ]/�ixսixսu�}�}󶽅���������7L��C���\)��hs������P���P������ �357BNP[grkg^[ONB<533T_dfnxz������zpa^WT��������������������_fkz���������zmaa]]_���
��������������������������������#/8<@D@<6/(#),6<BMGB<6)��������������������������������������� 	.6/���������.;HTaz�����{maH;2,).����( �����25BN[gmqqnlg^[NG?5,2������������������������������������������������������������   ��      269BFO[`[WSPODB<6.22��������������������W[hity�����~th_[USWW������������������������������������������������������������#$/<HMTNH</# #0<U{z�����{b<#���������������~~��5BNXg����������tOB<5��������������������#/2<HD=</#����������������������������������������QU]anqwxqnea_UPKQQQQ��������������������Wanz����}zwna\WWWWWW���uihh\US\hu�������������������������������������������������� �

���!'#/<HUaia\UH<3/# !")*56@BOOQOFB6/)""""����� ))�������)6B[s������th[OJ<+')��������}z������������������������������������������������������ �������z�����������������zz
)5BEKMKB5)
������

������dmz����������zmkfbad������ �������������[acnpqna`][[[[[[[[[[&0IUgqrqsnbUI<503*&&������������������������������������������������������������)5[gm_[\TNB6)
sv}�������������vzts08@KO[hv|{tpig[OEB70��������������������#0<IUkmkhXU<4/-'#w{��������~{wwwwwwww��������������������#06IUY[[YUI<2'!#&.0<AIKIJIFD<0$'#!����

��������������������������yz��������������zyxy���������������������������������������������������������������������������'1/)$����������������������������������������������������������������ltut����������tollll#'/3<EFBA<;/#��#%#
�������|�������������tlgkp|���������������������������������"*()%�����qt�������������tolnq;<EHLMHF@<999:;;;;;;�����������������������������������������������g�a�s�u�����������������������������������������žɾ�������������������������������������)�6�B�P�S�X�\�B�6�)������������������������������������������Z�T�M�E�G�M�V�f������ľǾþ�������s�Z�B�9�6�.�2�6�<�B�O�[�`�[�S�O�L�C�B�B�B�B�����������������������������������������"�����/�2�&�%�H�T�m�z�|�x�u�f�H�;�"àÞÓÍÇ�z�y�z�~ÇÓàìó����ÿùìà�`�]�������ʼ��!�.�E�H�@�2��ּ����r�`�������������������������!���	����������������������������� �	���	���������������������������������	���	�������׻v�l�h�c�e�l�x�������������»Ļ��������v�ƻƻлܻ����M�f�r������w�f�M������N�F�E�N�R�Z�g�s�w�z�|�s�g�Z�N�N�N�N�N�N�Z�V�P�S�Z�f�s�t�������s�f�Z�Z�Z�Z�Z�Z�_�X�S�I�F�?�?�F�J�S�_�l�n�x�z�x�v�l�_�_���������������ĿͿݿ����������ݿĿ����Y�M�U�Y�d�e�j�r�r�r�|�~�����~�}�r�e�Y�Y���������������	����"�&�(�$�"��	��������*�/�7�C�O�\�n�u�}�u�d�\�O�C�6�*��m�h�m�y�����������������������y�m�m�m�m���������(�-�.�0�/�(��������g�V�D�<�3��5�A�s��������������������ݿݿѿο˿Ϳѿݿ�������������ݾ�ԾϾ׾߾����	���"�1�:�:�4��	������׾ʾ��������ʾ��	�� �$���	����A�<�?�A�H�M�Z�f�g�l�m�f�Z�M�A�A�A�A�A�A�y�m�h�c�c�m�y�������������������������y�����������ĿſǿĿ����������������������A�=�4�1�0�4�A�M�Z�^�f�i�f�^�Z�M�A�A�A�A�L�K�J�L�M�Y�d�a�Z�Y�L�L�L�L�L�L�L�L�L�L�H�E�C�B�E�H�S�U�a�c�a�_�Y�U�H�H�H�H�H�H�"�.�5�7�;�;�;�8�.�"����"�"�"�"�"�"�"�6�6�8�*�����&�6�C�\�h�p�q�h�[�O�C�6ŭũŠşŠŧŭŰŹżŹűŭŭŭŭŭŭŭŭ�����ݿѿĿ��������Ŀѿݿ����� �������/�#�
������#�)�/�B�=�<�H�N�L�H�<�/�a�U�T�H�G�;�3�;�;�H�K�T�Y�]�a�f�a�a�a�a�T�;�+��.�;�T�m�y�����������������y�`�T���l�g�w�s�c�f�s����������������������²²®²¿��������¿²²²²²²²²²²��������������������������������������������!�-�.�0�-�!��������������}�����������������������������4�%� �$�(�5�A�N�Z�g�g�s�~�z�z�v�q�Z�6�4�Z�T�N�J�H�J�T�Z�g�s�����������}�v�s�g�Z�Ⱥ������������ɺ�������#����ֺ�ƳƪƧƪƭƳ��������������������������Ƴ������������$�.�4�5�2�0�$����������̿����������ĿƿƿĿ����������������������!����!�-�:�F�S�\�l�y���{�l�_�F�:�-�!�n�c�a�[�_�a�n�zÓÜàêæáÓÑÇ��z�n�����������������½ĽƽнѽսнǽĽ�����ŠŔŔŊŔŞŠŭŴŹ��ŹŭŨŠŠŠŠŠŠż�������������������*�/�(���������żŹŭŠŔŊŇŀŇŒŠŹ��������������żŹ����нͽȽʽнݽ����(�.�-�(�$��������(�5�A�L�A�5�/�(������������������ûл�����������ܻлû��û����������ûлܻڻлȻûûûûûûûûS�L�K�S�_�f�l�x���x�l�_�S�S�S�S�S�S�S�S�ݽڽݽ޽��������(�0�<�=�4�(����ݽ������������������������Ľнܽ۽нĽ��������������������������������������������������������������ûлܻ��ܻػлû������������������������ù͹ϹڹϹù���������������'�3�@�L�W�L�H�@�3�'�����e�\�e�f�e�e�e�r�~������~�r�e�e�e�e�e�e��ּмӼݼ��������������������üüûùþ������������������������B�;�6�)��)�2�B�F�O�[�h�n�tĀ�~�t�h�[�BĿļļĿ������������������������ĿĿĿĿ�����	�����!������������4�+�1�4�@�M�Y�\�Y�U�M�@�4�4�4�4�4�4�4�4ĦěĚčćčĖĚĢĦĳĵĻļĸĳĦĦĦĦE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�D�D�D�D�D�D�D�D�EEEE*E6E1E*EEED�D��
�����������#�/�<�H�R�a�b�W�U�H�<�/�
�y�s�l�a�l�n�y�����������������������y�y��
����$�%�$��������������������������!�.�:�B�G�Q�F�:�!������������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� ] P 7 C 9 < \ � I O ] - 7 + 6 \ 3 # _ 2 f U O � F ) K V V +  * # x g a : o _ Q � W > 2 ` \ 1 v 3 W , f ^ , d H H e - 5 y + @ e F B 9 2 - z O M   G ^ Z D 7 T G / < ] 0 / w    �  D  �  >  �  �  �  �  �  �     U  �  �  c  �  �  �  �    a  �  �    �  :  j  z  �  I  $  �  u  �  �  �  ]    �  �  �  [  Q  �  �  U  �  �  �  �  m  !  \  �  d  {  �  �  D  �  �  X  f  3  &    �  P  3  �  �  <  �  �  U  �  �  u  �  �    W  (  �  �;�o�D��;D���\)�49X�\)��C��49X�0 Žo��hs����ě������,1��h���㼬1������`B���o��/��j��/������y�#�@��\)����j��`B�ě��+��/�\)�o����P��`B�D�����������0 ŽD���#�
�L�ͽH�9�e`B��\)�#�
�u��%�8Q�D���e`B�T���}�D���u�8Q�H�9����q���e`B���������O߽�o�O߽�������C���O߽�����hs�ȴ9��G���󶽶E���   �ȴ9��v�BNIB�BB4��B 
EB'JB!�B�B�^B"�B&B-^A��\BO�Bs�B D�B!�B;�B*<Bv�BD�B�B ��B�1B[�B�-B'!lB)�B	�BǤB%�B+�B-[�BRMB!,�B��B2o@B��B�B��B�BێBۈB��BB��BH�B�iB��B�/B"�4B �B��Ba.B&�B�uB!�dB��Bj�B
��BV�B�]B&�qB):IBa�B&L�B&�BuyB��B\�B��B��BٽB�B�B� B��B��B
^�B+B��B
�BׄB�/B�}B
��BòBÊB��B4��A�g�B:�B!��B(�B�FBHB�B-�^A�B>�B>uB ?B!AXBQ2B"�B8�B8'B�cB �B,~B7B��B'9@B)�"B	ĥB;�B�B+-9B-KtBF�B!�B �B2@UB�uB?;B�B��BE�B�8B��B?}B@�B?�BF�B4OB0�B#3�A���B��BE&B'�BC[B!��B��B��BB>�BȚB&K�B)G�B�B&��B&=;BR`B�BM�B��B��B��B��B��B4�B� B4B	�B@B�QB
��B��B?�B�MB
��B�<A��|A���AM��A�88AI��AF	GA���A��A��WA�_�A�A���A���A�S�@��Q@�TkA�.AA�@��7A{ ?�%A�v�B ��Ao��A���A�0DA�XAZ��AU�1A>D�An�CAv�VA<��?�IA��qA_��B �[A��xA|U!A��A��Ai9�A�˳A�ŬA�Ρ@c�>@���A�۹A��>@GB�B�BպAv��@��A���A%<{A��A��A��7A.q(A�^�@�׼@���@�xVA4'�A$NA��?@�x�<1�?���?�L�Aj�A���A��tA�`@�E�@��gA��BC�[CC�Z3A�(AسB	�AU7B�JC���A�y�A���AM(�A��@AI-�AGAA�wA�5tA��MA�{(A�$A��A��sA�t�@�@�=�A�X�AB��@��TA{�?�ctA���B ʊAm�A��^A�O#A�*AZ��AT��A>�oAn��AwFA;�I?�A�xgA`�7BqA���A~��A�|�A���Ai�A�p�A�A�j�@c�@�W�A��A���@5�B8�B	7�Av��@siA�z�A$��A��A��GA�ÂA.|\A�yw@��?@�$@��]A5�A# �A�`�@��C��?�0�?�Z\Ar�AЀ�Aڠ�A䈳@���@��fA��DC�W�C�X�A��mA�B	=MAJB,�C��            )         	      "      =   6            i      	               	      	   :      *                              	               2                           #            	   	                                             O   (   !                  )   6      	   4                  %      #         +      ?   )            -                              7      %   !                                       #   1               !      %                        %            !                                                         !                           !                     5   !                                          5                                                #   1               !      %                        %            !                                                                        N�ߟOU	IN8B�O��eN+V<O�RN��FN5��O�%�OG��PM�O��N�L�OE0XOI;Od��NН�N�?�N��EN��N�[�O��ORm�N�p�N͘LPy�(O�XO"=�O>��N��_N�!�N��NɎ�N�oN0-N�J�O5�_N5�Nƽ�OAtNe'AO��P6��N:�N���N��=N�=.O�O�O6��O��!OV^�O=��M��O��+O�O��N}e�O���Oac�O
��N]rJO�sN/~NA�9O ��N�P�N�4N���NۊwNo�=N�)�O(��O�.�O[�kN���NlNe� N��ODYO,�!O�g!N�]�N�mOV]OaU�N?$�  �     �  �  �  �  K  �  �  C  r  <    �  �  h    =  �  |  �  �  �  .  �  N  4  �  �  @  5  e  �  E  �  3  Q  �  a  �  `  �  U  �  �  �  U  *  �    �  �  &  h  	  A  I  �  F  �  N  �  !  �  �  �  �  �  u  B  �  2  	�  �  <  �     �  	/  
#  	  �  Z  �  w  �<e`B<D��;�`B;D���ě��#�
�t��t���t��#�
���
������o�e`B��C��aG��T���T���u���
��C���t����㼛�㼛�㼣�
�����w��/��1��j���
��1��1���ͼě����ͼě��ě��ě��ě����ͼ�����/��`B��`B�C���h����h��P�'t��t���w��P�#�
�#�
�#�
�D���'0 Ž0 Ž8Q�Y��<j�L�ͽe`B�e`B�u�ixս���}�}󶽅���������7L��\)���P���T������P������� �357BNP[grkg^[ONB<533T_dfnxz������zpa^WT��������������������`glz����������zmb^^`���
��������������������������������#/8<@D@<6/(#),6<BMGB<6)�������������������������������������������)-%���������/4;HTmy���~maTH;40/�����������6BN[ginnjigd[XNJB866������������������������������������������������������������   ��      269BFO[`[WSPODB<6.22��������������������X[ghtv���zthb[VSXXXX������������������������������������������������������������#$/<HMTNH</# #0<Unux�����{b<#���������������~~��W[gtx�������tqg^[TPW�������������������� #/1<EB<;/#      ����������������������������������������QU]anqwxqnea_UPKQQQQ��������������������`anz����znaa````````���uihh\US\hu�������������������������������������������������� �

���!'#/<HUaia\UH<3/# !")*56@BOOQOFB6/)""""����� ))�������)6B[s������th[OJ<+')��������}z��������������������������������������������������������������z�����������������zz)5BDJKHB5)!������

������emz�����������zmgcbe��������������������[acnpqna`][[[[[[[[[[&0IUgqrqsnbUI<503*&&������������������������������������������������������������)5[gm_[\TNB6)
sv}�������������vztsNOY[htuvtqmhf[OIHJIN��������������������#0<IUkmkhXU<4/-'#w{��������~{wwwwwwww��������������������##%0<INUWXUKI<900(##"#'/0<@IKHHEB<0&)%#"����

��������������������������~����������������~~���������������������������������������������������������������������������'1/)$����������������������������������������������������������������ltut����������tollll #(/<DCB@<9/)#����

������p��������������tpp���������������������������������%%# ������qt�������������tolnq;<EHLMHF@<999:;;;;;;�����������������������������������������������g�a�s�u�����������������������������������������žɾ�������������������������������������)�6�<�B�N�P�O�B�6�)������������������������������������������f�c�Z�M�I�M�Z�f��������ľ���������s�f�B�9�6�.�2�6�<�B�O�[�`�[�S�O�L�C�B�B�B�B�����������������������������������������"����/�;�@�H�T�a�m�v�v�q�m�[�H�;�/�"àÞÓÍÇ�z�y�z�~ÇÓàìó����ÿùìà��v������ּ��!�.�?�B�:�.���ּ���������������������������������������������������������	�
�	������������������������������������������	��	�������׻��x�p�f�i�l�x���������������������������� ������'�4�@�M�Y�e�j�g�Y�4�'����N�F�E�N�R�Z�g�s�w�z�|�s�g�Z�N�N�N�N�N�N�Z�V�P�S�Z�f�s�t�������s�f�Z�Z�Z�Z�Z�Z�_�X�S�I�F�?�?�F�J�S�_�l�n�x�z�x�v�l�_�_�Ŀ����������Ŀѿݿ�����ݿѿĿĿĿĺY�P�Y�Y�e�e�l�r�z�~�����~�|�r�e�Y�Y�Y�Y���������������	����"�&�(�$�"��	��������*�/�7�C�O�\�n�u�}�u�d�\�O�C�6�*��m�h�m�y�����������������������y�m�m�m�m���������(�-�.�0�/�(���������W�F�=�4�(�(�5�A�s���������������������ݿݿѿο˿Ϳѿݿ�������������ݾ��������	���"�&�#�"����	�������׾Ͼʾ������Ⱦ׾�����	����	���A�=�@�A�J�M�Z�f�f�k�l�f�Z�M�A�A�A�A�A�A�m�l�i�m�u�y�������������������y�m�m�m�m�����������ĿſǿĿ����������������������A�=�4�1�0�4�A�M�Z�^�f�i�f�^�Z�M�A�A�A�A�L�K�J�L�M�Y�d�a�Z�Y�L�L�L�L�L�L�L�L�L�L�H�H�E�D�H�H�I�U�_�\�U�U�H�H�H�H�H�H�H�H�"�.�5�7�;�;�;�8�.�"����"�"�"�"�"�"�"�C�6�*�����(�*�6�C�\�]�h�m�o�h�W�O�CŭũŠşŠŧŭŰŹżŹűŭŭŭŭŭŭŭŭ�����ݿѿĿ��������Ŀѿݿ����� �������/�#�
������#�)�/�B�=�<�H�N�L�H�<�/�a�U�T�H�G�;�3�;�;�H�K�T�Y�]�a�f�a�a�a�a�T�;�+��.�;�T�m�y�����������������y�`�T���l�g�w�s�c�f�s����������������������²²®²¿��������¿²²²²²²²²²²��������������������������������������������!�-�.�0�-�!���������������������������������������������4�%� �$�(�5�A�N�Z�g�g�s�~�z�z�v�q�Z�6�4�Z�X�N�L�J�L�V�Z�g�s�����������z�t�s�g�Z�Ⱥ������������ɺ�������#����ֺ�ƳƫƨƫƮƳƷ������������������������Ƴ����������������$�*�0�2�3�0�$������快���������ĿƿƿĿ����������������������!����!�-�:�F�S�\�l�y���{�l�_�F�:�-�!�n�e�a�]�a�n�z�}ÇÓàäààÓÏÇ�}�z�n�����������������½ĽƽнѽսнǽĽ�����ŠŔŔŊŔŞŠŭŴŹ��ŹŭŨŠŠŠŠŠŠż�������������������*�/�(���������żŹŭŠŔŊŇŀŇŒŠŹ��������������żŹ�ݽڽнϽнܽݽ���������������������(�5�A�L�A�5�/�(������������������ûл�����������ܻлû��û����������ûлܻڻлȻûûûûûûûûS�L�K�S�_�f�l�x���x�l�_�S�S�S�S�S�S�S�S�����������
���(�4�6�7�4�(�'��������������������������ĽнٽٽнĽ����������������������������������������������������������������ûͻлܻݻܻػջлû����������������������ùȹϹйϹùù����������'�3�@�K�C�@�3�'���������e�\�e�f�e�e�e�r�~������~�r�e�e�e�e�e�e���ּԼּ�������������������üüûùþ������������������������B�;�6�)��)�2�B�F�O�[�h�n�tĀ�~�t�h�[�BĿļļĿ������������������������ĿĿĿĿ�����	�����!������������4�+�1�4�@�M�Y�\�Y�U�M�@�4�4�4�4�4�4�4�4ĦěĚčćčĖĚĢĦĳĵĻļĸĳĦĦĦĦE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�D�D�D�D�D�D�D�D�D�EEEE*E4E/E*EEED��<�
����������
���#�/�8�J�Z�\�U�L�H�<�y�s�l�a�l�n�y�����������������������y�y��
����$�%�$���������������������������!�.�9�:�D�<�:�.�!������������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� ] P 7 @ 9 ; \ � T O S  ) + : K 3 # _ - Y U O � F ' K F Q '  * # x ` a < o _ Q � W > 2 ` \ % v / W ) @ ^ , ` H H e - 7 y + @ e 2 A 9  ( l O @   G ^ Z D 7 N > ( < ] , / w    �  D  �  >  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  a  �  �    �  :  f  �  �  �  $  �  u  �  �  �  ]    �  �  �  [  Q  �  �  �  �  �  �  �  �  !  \  ;  d  {  �  �  1  �  �  X  f  "        �  �  �  z  <  �  �  U  �  �  C  �  [    W  �  �  �  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  @^  �  �  �  �  �  �  v  f  W  H  ;  1  &      �  �  |  I            �  �  �  �  �  �  �  �  �  �  �  o  S  ^  ]  F  +  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  Z  �  �  �  �  �  �  �  p  A    �  Y  �  m  �  W  �  �  �  �  �  �  �  �  �  �  �  w  l  a  V  L  C  :  0  *  %       u  �  �  �  z  m  \  G  -    �  �  }  4  �  �  v  U  $  �  K  :  *    
  �  �  �  �  �  �  �  �  �  �  {  d  I  %    �  �  �  �  �  �  �  �  �  �  �  �    z  t  o  i  d  ^  Y  i  x  |  �  �  �  �  �  m  G    �  �  �  �  �  �  j  #  �  C  (    �  �  �  �  o  h  ^  A    �  �  �  X    �  �  L  W  c  `  r  `  ;    �  �  j  -  �  �  b    �  W  �  x  �  �  �    !  2  <  2  $    �  �  T  �  �  6  �  5  �    M  �  �  �  �          
  �  �  �  �  �  p  1  �  y     �  k  �  �  �  �  �  }  p  `  R  B  0      �  �  �  f     �  �  �  �  �  �  �  �  �  a  /  �  �  z  6    �  �  �  V  �  s    �  �  
  
�    V  h  X  4    
�  
�  
'  	�  �      �              �  �  �  �  �  �  �  �  �  �  t  K  !   �  =  *              �  �  �  �  v  [  @    �  �  g    �  �  �  �  �  �  �  ~  n  W  3    �  �  x  N  B  Y    �  <  N  _  n  u  y  {  |  {  x  u  q  k  a  R  9    �  S   �  �  �  �  �  �  }  d  J  1      �  �  �  �  �  E    �  �  �  �  �  �  �  l  R  7    �  �  �  u  I    �  �  �  �  �  �  �  �  �  �  �  �  �  z  c  J  /    �  �  �  �  �  n  %  .  +  '  #       #  )  .  3  2  ,  &         �  �  �  �  �  �  �  �  �  �  }  v  m  d  S  :  !    �  �  �  P     �  7  @    �  �  �  V    �  �  ]    �  �  F    �  >  �   �  4  ,  %         �  �  �  �  �  z  ^  =    �  �  �  O      2  M  i  �  �  �  �  �  �  �  �  g    �  u    9  H  A  s  �  �  �  �  �  �  �  �  �  �  c  8  �  �  i    �  �  8  @  @  >  :  4  *    �  �  �  �    [  4    �  �  �  8  �    &  /  2  4  4  4  /  (      �  �  �  �  �  �  �  q  H  e  c  `  ^  \  Y  W  R  L  G  A  ;  5  -  "         �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  f  U  C  0    E  A  <  7  3  .  )  "      	    �  �  �  �  �  �  �  �  s    �  �  �  �  �  �  �  �  �  �  �  �  �  r  T  8  ;  J  3  .  )  #        
   �   �   �   �   �   �   �   �   �   �   |   m  9  G  N  F  F  M  ?  ,    �  �  �  �  y  M    �  �  f  F  �  �  �  �  �  �  �  �  d  7    �  �  p  <    �  �  ^  $  a  [  T  L  @  3  $      �  �  �  �  �  �  v  j  P  1    �  �  �  �  w  k  c  J  /    �  �  �  �        �  �  �  `  \  W  S  N  F  4  !    �  �  �  �  �  �  �    f  M  3  �  �  �  �  �  �  u  a  K  -    �  �  �  g  3    �  �    U  I  @  ?  >  5  6  -    �  �  �  \    �  �  n  !  �    �  �  �  �  �  �    z  u  p  k  f  a  \  W  V  W  X  Y  Z  �  �  �  �  �  �  �  �  �  o  O  (          �  �  P    �  �  �  }  l  W  7    �  �  E  �  �  H  �  �  *  �  Y   �  )  6  B  J  O  S  T  M  7    �  �  �  r  M  8  I  �  �  !  *            �  �  �  �  �  |  S  )  �  �  �  �  `  :  �  �  �  �  �  �  �  �  �  p  S  4    �  �  Z  �  �  3  �        �  �  �  �  �  �  �  }  T  -    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  e  >    �  �  n    �  [  �  n  z  �  j  �  �  �  }  X  %  �  �  S  �  �  -  �    a  �  �  &  "                      �  �  �  �  �  �  �  �  h  V  @  0  *  ,  2  @  <  /  !    �  �  |  B  	  �  �  s  �  	  �  �  �  �  s  <     �  �  p  )  �  X  �  ]  	    �  A  <  7  0  (       
  �  �  �  �  �  �  �  }  \  4      �  I  =  0  !    �  �  �  �  �  u  P  +    �  �  �  T    �  �  �  �  �  t  _  H  )  /  2  .  &      �  �  �  l  �  }  F  >  7  1  *  $           �  �  �  �  �  U  $  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  ?    �  �  ~  l  g  N  7     	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    g  N  4    �  �  �  �  }  d  c  !               �   �   �   �   �   �   �   �   �   �   �   �   �  �  �  �  �  �  �    s  g  [  O  E  :  /  $        �  �  ,  o  �  �  �  �  �  �  �  d  B    �  �  �  I  �  a  �  r  �  �  �  }  w  g  [  O  :  !    �  �  �  �  �  j  E  <  l  �  �  �  �  �  �  �  �  �  �  ~  m  [  G  ,    �  �  �  �  �  �  �  �  �  �  �  �  �  �  `  .  �  �  G  �  g  �  -  C  p  q  r  t  p  h  \  N  8    �  �  �  o  >    �  �  }  K    #  +  :  ?  >  1  #       �  �  �  o  =    �  �  H    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  �    2  &  �  �  e    �  P  �  `  �  �  
�  	�  Y  �  Y  	�  	�  	h  	C  	  �  �  a    �  y     �  `  �  �  �  a  �  �  �  �  �  �  �  �  �  z  S  )  �  �  �  O  �  �  ;  �  �    <  5  /  )  "          �  �  �  �  �  �  �  �  o  Y  D  �  �  �  �  y  o  e  Z  O  E  8  )      �  �  �  �  �  �       �  �  �  �  j  H  $  �  �  �  �  }  _  A    �  4  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  	  	-  	  �  �  �  r  A    �  u  	  �  #  �  :  �  �  �  q  
  
  
  	�  	�  	U  	  �  E  �  4  �  P  �  �  �  �  �  H  �  �  	   	  	  	  	  �  �  |  0  �  �  ,  �  _  �  >  �  �  �  �  �  �  �  �  o  Q  .    �  �  |  S  (  �  �  �  m  (  �  Z  a  h  S  -    �  �  �  d  ;    �  �  �  �  d  J  0    P  j  ~  �  �  }  t  `  B    �  g  �  j  �  �  E  �  Q  �  w  e  L  ;  (       �  �  �  r  6  �  �  c    �  m    {  �  �  �  q  a  Q  >  *    
  �  �      !  ;  U  |  �  �