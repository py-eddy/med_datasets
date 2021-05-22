CDF       
      obs    I   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�;dZ�     $  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�*�   max       P9)     $  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��j   max       <�o     $  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?E�Q�   max       @F�G�z�     h  !   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @v}G�z�     h  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @P            �  7�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�         max       @�3`         $  8|   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �o   max       ��o     $  9�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�'   max       B4�R     $  :�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B4|D     $  ;�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =�-m   max       C�΢     $  =   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       > [(   max       C��"     $  >0   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          >     $  ?T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7     $  @x   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5     $  A�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�*�   max       P,m>     $  B�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��*0U2b   max       ?х�oiDg     $  C�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��j   max       ;ě�     $  E   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?E�Q�   max       @F�G�z�     h  F,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @v}G�z�     h  Q�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @P            �  \�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�         max       @�`         $  ]�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E5   max         E5     $  ^�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�	� �   max       ?х�oiDg     �  _�                               
            .   2                                 
   )            2   "   
            =   3            7                        /                        	   ;   0                  :         
            O��\N�(�N`N�Q@Nu!�P,i�O�d�O��M�*�O0��N�#�N�4NH|�Oc�O�#�OWPPNt-�NR��O���N���Or0N�ײNZf�N+X�NV�N�O�h4N�J�O���NUTEO��O�eOMݒP�`N[�OT3\P9)O�4IN���O��O�(WO� �N�^�N��O��OsUO�U�O"�N���P1�$N`Y;O���OQ�O�-3N�h�O�2�O6��Nu,6O�HKO�G�O+$N0eSN6�UO�E�N,,�O}ҋN7aNf�NͰqN?[{N�'N.- O<q<�o;ě�;D��;o;o;o:�o%   ��o�D���D�����
�o�o�#�
�49X�D���e`B�u��o��C���C����㼬1������/��/��/��`B��`B��h���o�+�+�+�\)�\)�\)�t��t��t��t���P���#�
�,1�,1�8Q�<j�@��@��@��H�9�L�ͽaG��aG��m�h�m�h�q���y�#��o�����7L��C���hs���㽛�㽝�-��{��-��j��j���
#)5::0"
�������

��������>HUafiaUH>>>>>>>>>>>NPX[gt|����tslg`[UNN��������������������{��������������yx{x{��������������������6KOY[hsh[OI6������������������������������������������������������������
#*'# 
	CHU[]YUHEBCCCCCCCCCC).6BDORVX]`a[OB6.'&)��������������������<BFNT[gty����tg[NB7<@HU[ajhaaUH@@@@@@@@@'*568CDFC?6+*$''''''��������������������������������������.<ania\asmaYRH</..+.��������������������������������������������������������������������������������8<BHTSIH<;8888888888"'0<IU\^`b_XUS</(#!"FILTUW^bbccbZUIA?AFF����������������������

��������������!#*##
�������)BOh�����}qh[O6%
#0?IUY^`YL<0#

7@HJmz�����{mYJHC;77:;HIKHB;;8::::::::::�����	��������9NV]alcB5)����������������������������������������")5[bb^cg|sgNB/&&(!"gqst���������todabdgn��������������tmkkn��������������������3<HUU\UH@<3333333333����������������������������������������moz��������������zom��������������������ggt����������tgfgggg������%$��������~���������{v~~~~~~~~��������������������&6;BO[hkmmqsmh[O6)%&���
#(0:?=90"
���mmruz}�������zpmmmmmN[gt���������t[NKIJNstw|��������������ws�������������������������������������������
#-.,(),.#
����cgot���������tlgca_c��������������������	��
������������������������������������������7<HHKIH@<<;976777777DHMUaaeaUIHFDDDDDDDD���������

��������������������������"#/46/*# """"""""""./<=BHIKH<:6//......OUanz��}zna_XUSOOOOàÃÀÉÏÐÓàìù��������������ùìà�����������������Ƽʼμʼ����������������A�9�9�>�A�M�R�W�R�M�A�A�A�A�A�A�A�A�A�A���������������������������Ŀ˿ѿԿѿǿ��y�n�r�y�������������y�y�y�y�y�y�y�y�y�y���������~�|�����������������ʾ��������������̽нݽ���+�.�(�����ݽ�������
��"�1�;�C�G�H�K�J�L�T�h�l�a�H�;�/�����ùøù�������������������������������z�n�f�a�U�Q�K�U�a�n�zÇÌÓÙØØÓÇ�z�����#�,�/�<�F�D�<�;�/�#�������������������������������������������������������������������������������������ѻD�:�2�6�:�B�F�S�_�x�������������x�l�_�D��������������$�0�=�G�I�G�@�0�$����	��������������	�� �!�����	�/�&�&�"��"�/�;�;�@�D�;�/�/�/�/�/�/�/�/�	���������	���"�%�"�!��	�	�	�	�	�	��r�Y�R�R�S�P�S�Y�f�r������������������������������!�!�!� ��������
�� ��#�*�<�D�H�U�W�U�H�B�D�>�<�#��
����������������������������������Ѿ�����������������������������������������޽ݽнͽнݽ����������������麰���������ɺֺݺںֺɺ������������������6�.�)�(�)�6�B�B�J�B�6�6�6�6�6�6�6�6�6�6�������|����������ݽ�������ݽнĽ������(����	���%�(�4�A�C�M�Q�W�M�A�4�(�(�	��������߾����	��.�9�=�5�.�"��	�ʾǾ��������ʾ׾׾��׾ʾʾʾʾʾʾʾʿ������������������Ŀѿ����������ݿĿ����y�|�z�l�d�g�s�����������������������������������������������������������������5������޿ٿݿ���5�N�g�������t�N�5������������������������������������������	��������"�.�;�G�T�T�O�S�T�U�G�.��z�n�i�a�V�aÇàù������������ùìÓÇ�z�l�Y�O�B�Q�Y�������������������������l�L�K�L�N�L�I�L�Y�e�r�w�}�u�r�f�e�e�Y�L�L��������������*�9�C�O�T�S�O�C�6�������x�b�U�I�=�I�U�n�{ŔšŭſŽŹŭŠŔŇ�x�B�(�"�!�$�,�6�B�O�[�h�tćčċ�}�h�[�O�B���������������$������������������������������������������������������������������������*�,�,�*������x�t�n�r�x���������ûлܻػлû�������ƳƧ��p�d�O�=�6�*�:�C�P�\�uƚƧƶ����Ƴ�������������������)�!����������������������������������������������������`�b�r�������ֽ���%�'� ���ּ������`���������������������������������������޺�ֺ̺ܺκֺ������#�-�:�B�:�!����ܹ׹Ϲʹ������ùϹܹ�����������ܻܻٻͻû������������ûлܻ����������������������������������������������²¨©¨­¬¬²¿����������������¿²�<�/�#���������/�<�H�U�[�\�U�H�<�5�,�(�'�(�(�5�A�N�Z�]�Z�N�J�A�A�5�5�5�5�ӻû����������л����)�/�*��������ED�D�D�D�D�D�D�EEE7ECEPE[EZEPE7E*EEĿļĿ������������������������������Ŀ����������������������������������������������(�,�4�:�A�G�A�4�(�������*�����!�.�G�S�l�y�����������n�G�:�*�������������ʾ׾۾׾ʾ����������������������������������Ŀѿݿ������ݿѿĿ�D�D~D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�E�E�E�E}E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��0�)�$��������� �$�)�0�3�2�5�2�0�0�/�&�"����"�/�;�H�I�J�H�=�;�9�/�/�/�/E�E�E�E�E�E�E�FE�E�E�E�E�E�E�E�E�E�E�E�FFFFF$F0F1F=F?F?F=F1F$F$FFFFFF���������������ùϹع۹ӹϹù����������� : " Y n ? 6 x O h R T & + K  6 Z ! / # M   : 6 = H J E 8 6  e p z a < ) ( 6 T h  2 r ; F n T { ` = K Q C ; 9 4 ` P L B Y T p p  x V J � Q m .    �  �  �    o    ^  {  7  �  �  �  X  �  �  �  �  _    �    �  [  I  s  3  |  �    ^  �  7    �     �  #  �  �  Y  l  �  �  x  H    �  G  	  �  f  ]  �  L  �  �  �  �  d    x  d  P  �  ^  �  �  L    �  !  z  6�D����o���
�D���o��h�u��t����
�T����`B�#�
�D���Y��u���e`B��t��8Q켬1�C���`B�ě��+�o�t���7L���m�h�\)���w����'q���t��L�ͽě��� Ž��y�#�D����^5�0 Ž�w�P�`��C��m�h�L�ͽ@���vɽL�ͽ��P��7L��7L�m�h���㽟�w��+�������㽉7L��\)�Ƨ𽗍P�o��9X���-��j���`�ȴ9��G�B�B$g�B��B	N�B�B U�B"Q�BmB �sB��B1XB$�B�B:FB�B	�B��B05�B �B"��B��Ba5B4�RB)��B!w�B&�B&�yB'7Bk�B�B�B�B&@]A�7)A�'B��B�YB�1B!P�B�B
 ?B]B*EBB#B �WB �:B��B
<\B-0�B
��Bv\B�yB$�GB �B	��B
��B��B��B�(B
8B�*BY�BΠBl�B��B��B��B��B��B�B�(B�B��B$C�B��B	1{B�B (#B"@B8�B ��BP�B(�B$�OBB�BBAB��B	8 B�_B0:'B�tB"�B,BI�B4|DB)�aB!PB;�B&C	B&�CB@%B��B8�B��B&�A�w�A��B�;B?rB�5B!uOB�WB
2PB?�B*�B@�B?9B �xB @-B�bB
m�B->vB
��B�iB'hB$�=A�sLB	��B
�B��B�B�ZB
9-BX�B?EBIrB@1B�B��B�\BCB�lB��B��B�A�@��A;��Av�.AoNpAP~AA*�A���A�A�~�A��AH;A�i-@�ގB	|�AZA��lA\g�@��@_P�A��KA�؞AK�A,8@17KAט�A$��A9G�A[��AQ߂Ay��A��A�̷A��A�C-A_�uAˎg@�e�?� .A���A��AٶA��@A�ӷA�@�@�VyB �A��6A�V@�{RA�S9@Yt�?�@�Q�A�HkA��A�^fA�ed@���C�|4A�*A��	A7ۀA��AO��Ay�C��C�?B	�mA�}C���C�΢=�-mA�z&@�vA<�GAv�)Apu�AP��A)
�A�J�A�hDA��A�}�AH:�A�|�@��B	�AAY�VA��'A[g@��@^8VA�{A�rNAK�tA,{F@.<�A�}~A!GwA:�`A[M�AQ;Az�A�utA��A�~A�u:A_��Aʀ@��?���A�y�A��A��A�~4A��A�x�@��B>DA��A�]�A�FA���@M�J>��M@�P�A���A�)-AÚA���@��C�]�A��A�w�A7,AdAN��Ay9�C��[C�TB	6�A�{�C��aC��"> [(         	                                  .   3                                 
   )            2   #   
            >   4            8                  	      0               	         	   <   0                  :                        #               +   '                                                            !               #      1         +   !      #                     )         7            !               '               #                                             #   %                                                                                 1         %                              )         5            !               #               !                           O!�N�(�N`N�Q@Nu!�O��kO�OROn��M�*�N���N�#�N�4NMN�t�On�VN�F�Nt-�NR��OM�lN���Or0N_��NZf�N+X�NV�N�O�muN���OW�dNUTEO��'O
إN�P�`N[�N�W�P��O�»N���O��ZO�(WO��N�^�N��N�?O$ρO�U�O"�N���P,m>N`Y;O���OQ�O�-3N�h�O�2�Nø?Nu,6OÈ=Or��O+$N0eSN6�UO�IN,,�Os�N7aNf�NͰqN?[{N�'N.- O<q  W    )  z    6  <  K  �  �  �    �  s  �    �  �    �  �  4  �  ]  N  �  �  �  �    E  �  �  �    �      �  �  E  �  j  /    �  �  !  S  �  �  W  Z  �  �    *  �  x  
�  J    ]  �  B  	�    ^  5    �  H  ;D��;ě�;D��;o;o�t���o�o��o���
�D�����
�t���j��j��t��D���e`B���
��o��C����
���㼬1������/��h��`B����`B�#�
�8Q�\)�+�+���@��'\)�#�
�t��L�ͽt���P�#�
�<j�,1�,1�8Q�@��@��@��@��H�9�L�ͽaG��}�m�h�}󶽇+�y�#��o�����C���C���9X���㽛�㽝�-��{��-��j��j���
#(,/1/)#
�������

��������>HUafiaUH>>>>>>>>>>>NPX[gt|����tslg`[UNN����������������������������������������������������������)6FOSQOFB;6������������������������������������������������������������
#*'# 
	EHUXZVUHHCEEEEEEEEEE56BGOQSUUQOB@64.+.55��������������������KNS[gntzvtg[XNKKKKK@HU[ajhaaUH@@@@@@@@@'*568CDFC?6+*$''''''����������������������������������������.<ania\asmaYRH</..+.��������������������������������������������������������������������������������8<BHTSIH<;8888888888#)1<IUZ\]_`a]U<0)$"#FIMU\abbbbYUIB@BFFFF����������������������

�������������

���������HOZ[ht|{trha[VOGCHH!#&09<IUUYUQIB<0'#!!7@HJmz�����{mYJHC;77:;HIKHB;;8::::::::::���������������5BNUUZ^XB5)��������������������������������������&+5BNYZYZ^b[NB2((,,&gqst���������todabdgru��������������utrr��������������������3<HUU\UH@<3333333333����������������������������������������moz��������������zom��������������������ggt����������tgfgggg���������%$�����~���������{v~~~~~~~~��������������������&6;BO[hkmmqsmh[O6)%&���
#(0:?=90"
���mmruz}�������zpmmmmmN[gt���������t[NKIJN������������}y{������������������������������������������
""!$%()#
����cgot���������tlgca_c��������������������	��
����������������������������������������������7<HHKIH@<<;976777777DHMUaaeaUIHFDDDDDDDD���������

��������������������������"#/46/*# """"""""""./<=BHIKH<:6//......OUanz��}zna_XUSOOOOìáßßÚÛàìôùû������������ÿùì�����������������Ƽʼμʼ����������������A�9�9�>�A�M�R�W�R�M�A�A�A�A�A�A�A�A�A�A���������������������������Ŀ˿ѿԿѿǿ��y�n�r�y�������������y�y�y�y�y�y�y�y�y�y��׾ʾ��������������������׾��� ����㽞�����������ݽ����(�*������нĽ��������"�7�;�H�H�T�W�a�f�i�a�T�;�/�"�����ùøù�������������������������������z�w�n�i�b�a�]�a�n�zÇÉÓÖÕÓÒÇ�z�z�����#�,�/�<�F�D�<�;�/�#�������������������������������������������������������������������������������������ѻF�F�B�F�N�S�_�l�x���������{�x�l�_�S�F�F������������$�0�6�=�?�B�?�7�0�$������������������	�������	���������/�&�&�"��"�/�;�;�@�D�;�/�/�/�/�/�/�/�/�	���������	���"�%�"�!��	�	�	�	�	�	�Y�U�U�V�T�Y�f�r�����������������r�f�Y��������������!�!�!� ��������
�� ��#�*�<�D�H�U�W�U�H�B�D�>�<�#��
�������������������������������������������������������������������������������޽ݽнͽнݽ����������������麰���������ɺֺݺںֺɺ������������������6�.�)�(�)�6�B�B�J�B�6�6�6�6�6�6�6�6�6�6�������}�������������нݽ���۽ϽĽ������(�!����(�*�4�A�A�M�P�V�M�A�4�(�(�(�(��	�����������	��"�.�4�:�2�.�"��ʾǾ��������ʾ׾׾��׾ʾʾʾʾʾʾʾʿ������������Ŀѿݿ�������ݿѿĿ�����������x�w���������������������������������������������������������������������5������޿ٿݿ���5�N�g�������t�N�5������������������������������������������
�	��	���"�.�;�E�C�D�;�.�"����Ç�t�l�j�k�wÇÓìù����������÷ìàÓÇ��r�Y�M�O�U�g�������������������������L�K�L�N�L�I�L�Y�e�r�w�}�u�r�f�e�e�Y�L�L�����������������*�C�J�O�L�C�6������x�b�U�I�=�I�U�n�{ŔšŭſŽŹŭŠŔŇ�x�O�B�3�*�)�,�6�B�O�[�g�t�w�z�y�t�h�d�[�O���������������$���������������������������������������������������������������������'�*�*�*�!��������x�x�r�x�{�����������»�������������ƳƧ��p�d�O�=�6�*�:�C�P�\�uƚƧƶ����Ƴ�������������������)�!����������������������������������������������������������f�c�r�������ֽ���%�&����ּ����������������������������������������޺�ֺ̺ܺκֺ������#�-�:�B�:�!����ܹ׹Ϲʹ������ùϹܹ�����������ܻܻٻͻû������������ûлܻ����������������������������������������������²¨©¨­¬¬²¿����������������¿²�#� � �#�&�/�<�H�U�U�W�U�L�H�<�/�#�#�#�#�5�,�(�'�(�(�5�A�N�Z�]�Z�N�J�A�A�5�5�5�5�ػл��������л����%�,�)��������D�D�D�D�D�D�EEE*E7ECEPEVEWEPELE*EED�ĿļĿ������������������������������Ŀ����������������������������������������������(�,�4�:�A�G�A�4�(�������+�����!�.�G�S�l�y�����������l�G�:�+�������������ʾ׾۾׾ʾ������������������������������Ŀѿݿ������ݿѿĿ�����D�D~D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�E�E�E�E}E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��0�)�$��������� �$�)�0�3�2�5�2�0�0�/�&�"����"�/�;�H�I�J�H�=�;�9�/�/�/�/E�E�E�E�E�E�E�FE�E�E�E�E�E�E�E�E�E�E�E�FFFFF$F0F1F=F?F?F=F1F$F$FFFFFF���������������ùϹع۹ӹϹù����������� D " Y n ? 6 s ( h B T & , 2  1 Z ! , # M 8 : 6 = H F @ / 6  ? S z a 4 & " 6 Y h  2 r 5 + n T { ` = K Q C ; 9 . ` A W B Y T m p  x V J � Q m .    ~  �  �    o  �  �  �  7  !  �  �  "    �  �  �  _  �  �    y  [  I  s  3  -  �  �  ^  �  3    �       Y  �  �  �  l    �  x  �  c  �  G  	  �  f  ]  �  L  �  �  �  �  �  /  x  d  P  �  ^  F  �  L    �  !  z  6  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  E5  �  �  �        2  C  Q  R  1    �  �  �  T  �  �  �  |    �  �  �  �  �  �  �  �  �  s  a  O  =  .  #           )        �  �  �  �  �  �  �  �  v  ^  E    �  �  �  �  z  t  m  g  `  Y  S  L  @  +      �  �  �  �  �  \  .      z  v  q  m  h  c  _  Z  U  Q  M  I  F  B  4  "    �  �  �      "  '  ,  3  6  2  '      �  �  �  �  c    �  .  !  )  7  9  3  +       �  �  �  ~  P  +     �  �  K   �   M  �  7  J  K  E  <  0  "    �  �  �  �  l  ?    �  �  �  �  �  �  �  �  �  u  _  J  4    	  �  �  �  �  �  �  �  �  �  v  |  �  �  �  �  �  �  v  ]  C  &  
  �  �  �  �  b  �  �  �  �  �  �  �  �  �  �  �  �  x  e  Q  =  )    �  �  u  G                �  �  �  �  �  �  �  �  �  �  r  M  (  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  	  2  P  f  r  o  Z  3    �  �  +  �    @  3    �  �  4  _  |  �  �  �  �  p  L  #  �  �  w  $  �  �    	    �  �  �  �  �  �      �  �  �  �  �  l  8    �  �  9  \  �    ~  }  |  {  z  y  x  w  t  p  k  g  b  ^  Z  U  Q  L  �  �  �  �  �  �  �  �  �  �    v  l  b  X  M  C  8  -  "  �              �  �  �  V    �  �  T    �  E    '  �  �  �  �  �  �  �  �  �  |  r  g  V  A  ,            �  �  �  �  �  �  �  �  �  �  �  �  z  <  �  �  a  !  �  �        (  /  3  (         �  �  �  �  �  �  �  |  G    �  �  {  r  j  a  W  N  D  7  +        �  �  �  �  �  �  ]  [  V  N  B  3  !    �  �  �  �  �  c  @    �  �  �  �  N  G  @  9  -  !      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  b  L  5      �  s  �  d  ?    �  �  �  �  n  T  4    �  �  3  �  r    �  �  �  �  �  �  �  �  �  c  E  %    �  �  �  v  L        o  �  �  �  ~  p  b  T  B  +    �  �  �  b    �  �  g  �          �  �  �  �  �  �  �  �  �  �  �  }  z  �  �  �  �  
  *  =  E  ?  0    �  �  �  �  N    �  c  �  ~    c  �  �  �  �  �  �  �  �  �  �  �  �  �  a  $  �  k  �  \  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  b  W  L  �  �  �  �  q  V  <       �  �  �  f  A    �  �  ,  �   �        	    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  `  8    �  �  <  �  �  �  �      �  �  �  a    �  6  �  6  �  n  Y  �  �   �  �  �  �  �  �  �  �  o    �  g  �  �    �  �  C  �    �  �  �  �  �  �  �  �  �  �  �  w  l  `  W  U  R  O  M  J  G  f    �  �  �  �  i  F  "  �  �  �  r  �  y  F  �  �  �  *  E  6  &      �  �  �  �  �  �  �  �  t  U  0    �  �    5  n  �  �  �  �  �  �  �  Z  (  �  �    �  !    i  �   _  j  h  f  b  Y  O  D  8  -        �  �  �  �  �  �  n  T  /  +  '  #                  �  �  �  �  �  �  �  �            �  �  �  �  i  6  �  �  �  G     �  Y     �  �  �  �  �  �  �  �  �  �  �  S    �  q    �  �  �    �  �  �  �  �  h  E  =    �  �  �  �  �  �  �  k  R  7      !      	  �  �  �  �  �  �  �  �  j  P  5    �  �  �  S  S  M  H  B  =  7  2  ,  &  !  $  /  :  E  O  Z  e  p  {  �  �  �  }  q  p  F    �  �  g  1  �  �  Y  �  �  8  �  �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  W  T  N  B  4  .  ,    �  �  �  H    �  {  .  �  _  �   �  Z  L  2    �  �  �  �  �  t  M    �    <    �  �    r  �  �  �  �  �  �  �  �  �  |  g  I  %  �  �  �  z  4  �  �  �  �  |  q  g  ^  U  L  D  <  4  *  !      
      H  v             	  	     �  �  �  �  b  1  �  �  x  '  �  #  �  �      "  '  *  *  !    �  �  �  N  �  �  &  �  E  i  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  B  l  t  ^  8    <  M  (  �  �  e    �  6  �    2  �  H  
+  
b  
�  
�  
�  
�  
^  
0  	�  	�  	L  �  7  �  |  �  �  �  �  	  J  A  :  5  1  -  '  !      �  �  �  Q    �  �  V    �                              '  C  _  z  �  �  ]  ]  ]  ]  \  T  L  D  ;  .  "      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  g  6  �  �  �  S    �  �  _  ;  �  B     �  �  �  �  x  y  �  �  �  �  h  N  4    �  �  �  �  �  	  	E  	k  	�  	�  	�  	�  	n  	>  		  �  �    �  �  �    h  =      �  �  �  �  �  v  e  =    	  �  �  �  �  w  Q  2    ^  X  S  M  B  6  )  #  !          �  �  �  �  �  �  U  5    	  �  �  �  �  �  �  �  �  �  �  �  �    ?  o  �  �    �  �  �  �  y  ]  @  #    �  �  �  �  ]  +  �  �  �  N  �  �  �  �  �  �  �  �  �  �  �  �  �  �  i  K  -    �  �  H  9  )      �  �  �  �  o  E    �  �  �  `  1    �  �    �  �  �  �  j  H  $  �  �  �  }  L    �  �  o     �  �