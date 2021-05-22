CDF       
      obs    D   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�9XbM�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Mޭ�   max       P��k       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �?|�   max       =C�       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?c�
=p�   max       @F�p��
>     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v{�z�H     
�  +|   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P�           �  6   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��           6�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �I�^   max       ��o       7�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�4a   max       B5%.       8�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�p<   max       B4�c       9�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�a	   max       C���       :�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��   max       C���       ;�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          O       =   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =       >   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =       ?$   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Mޭ�   max       P��h       @4   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���n/   max       ?�5�Xy=�       AD   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �?|�   max       =C�       BT   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?}p��
>   max       @F�p��
>     
�  Cd   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�     max       @v{�z�H     
�  N   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @P�           �  X�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��            Y,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >�   max         >�       Z<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��s�g�   max       ?���@��     @  [L   >            !   O   ;                               !      )   	         $   
      '                                        5   1            	                     '                  $                              	   
No�O	A�O��N8��NN�P�u�Pm�N�aO���N��O��8Pjm�NZ�IO8��O�^3N�fAP<(�N^�PEL�N���Nf��O�'O��6OM�O�L�P$HO9�Ob��NƊMޭ�O*B\O��_NN�eOMٝPg��O.�N=NWW�P>++P��kN�y�No�N�UuN�JVN�0�Op�OY�*N��O���N�V O���N�}�N�N��%O6�NW�P"�OU�GNM�O��Oex�N���N��9N���N�V=NйN:ŚNE�f=C�<t�<t�;D����o���
�ě��ě��o�t��#�
�49X�49X�D���T���e`B�u�u��t���t���t����㼬1��1��j���ͼ��ͼ�/��/��/��`B��h��h��h�������o�o�C��C��C��\)�t���P�'0 Ž0 Ž49X�49X�8Q�H�9�H�9�T���Y��Y��Y��u�u��%��������O߽��㽛�㽛�㽧?|�	'&(										��������������������nptz�����������ztjjn� !��������������������������������)BHI<,)�����0O[gnt��������[5))$0��������������������(/<@EKHHPH:/#��
#%./62/#
���������������������������'/+/)������569BO[aa[OB655555555����������������������������
������������

��������Qamz���������zmTPHCQ��������������������Taz����������aYRNNT�������������������������������������������������  �����������������������
!#&'#



��������������������3BN[t������tgNE:3.-3��������������������7CEHUaenuz��{pnaUH07269BDOOOGB<622222222��������������������46BOQU[^[VOD;6126:<4BO[ht��������t[OB=<BT[ehitw}th[UTTTTTTTTR^ht��������}toh[QR	#7Db{����sf<0#
	�����������������~������������������������������������������������ $������������5BNVXP8(�������������������������NOZ[ahnmhf[YOKNNNNNN��������������������FHSUUafnz{znia\UKHFF��������������������<CHLU`ahgcc`_]UHA<:<����������������������������������������fms{�������������tgfxz�����������zvwwxx��������������������z{���������{vsstzzzz
� ())59ABHB5))!N[gtz������tgd\[UPLN����������������������� )/54&"�����	
#-0;:4-#
����,0<<DD=<<0%$,,,,,,,,cgt����������{tf][[cegt������������wh`_e��������������������#'/3752/#"!�����

�������������

�������������

  ��-/<CHIOH<;/.--------.5BNOPNB5*..........�������(�2�/�(�����������������#�/�7�7�<�@�?�<�/�#�����ۻлû������������л�����������ۺֺѺɺȺɺкֺ����ۺֺֺֺֺֺֺֺ��n�k�g�f�n�x�zÇÉÇÀ�z�n�n�n�n�n�n�n�n��������<�G�M�Y������ȼɼ�����f�4��y�`�K�?�T�Z�]�k�y�����ȿݿ޿ؿֿҿ����y�H�G�@�G�U�U�a�b�n�z�{�z�z�z�z�n�a�[�U�H�<�4�#���#�/�<�D�J�U�d�n�w�x�n�a�U�H�<²®°²¸¿������������������������¿²�Y�<�3�2�=�M�Y�f�p����������������r�f�Y�A�'�"��������N�s���������������s�g�A�H�F�;�2�5�6�;�H�K�M�Q�I�H�H�H�H�H�H�H�H�B�6�2�/�)�)�)�6�B�O�Z�[�b�e�l�h�[�O�E�BàÓËÇ�z�s�o�n�|ÃÓàì������ûùìà�����������������������������������������������������(�A�g�����������Z�5�FFFFF$F$F0F1F1F1F&F$FFFFFFFFķıĴĲĴĿ�����
�0�<�0�)����������ķ��������������������������������������޾f�d�_�Z�W�Z�f�s�t�����������x�s�l�f�f���������������������.�6�9�9�7�0�$��"��	��	���'�/�:�G�T�m�y�y�n�Z�H�/�"���������	��"�/�4�;�?�>�;�/�"�� �������{�r�{ŇŏŠŭ��������������ŹŭŠŌŇ�{���������ξ߾���	���'�&���	����ʾ��������ؾվ׾ܾ���	�������	���Ŀ������������������Ŀѿݿ����߿ѿľs�h�f�d�f�q�s�u�����z�s�s�s�s�s�s�s�s��������	�����	���������������������m�g�V�]�`�m�u�����������������������y�m�W�O�V�g�u�|�������������������������l�W�e�^�Y�V�Y�d�e�r�t�y�y�r�e�e�e�e�e�e�e�e���������������������������������������������t�Z�A�2�)�D�Z�s����������������������ݿҿɿƿ��Ŀѿݿ���	����$����侥���������������������������������������A�?�4�-�(�#�(�4�A�C�M�O�T�M�A�A�A�A�A�A�r�g�q���������ּ���(�*����ּ������r��ã�u�X�<�"��"�<�aÇù�������	�
�����Ż!���!�!�-�:�F�H�M�F�:�1�-�!�!�!�!�!�!�ù¹��������ùϹܹܹ߹ܹعϹùùùùù�ƧƞƚƐƒƚƧƲƳ��������ƳƧƧƧƧƧƧ�M�G�A�@�A�J�M�X�Y�Z�]�a�f�i�f�c�Z�U�M�M����������������������������������������������ʾ������ʾ׾����	�����	���e�^�Y�V�T�S�X�e�r�~���������������~�r�e�����������ɺֺٺ�����ֺ˺ɺ�����������ŭŠŔŊŁŇŔŠŹ�����������������Ҿ��������������������ʾξоʾþ���������� ����� ���(�A�g�s�������g�Z�N�5��� ��
�
���'�*�4�@�B�K�G�@�4�'������`�]�`�f�m�y�y�������y�m�`�`�`�`�`�`�`�`�t�t�h�h�h�o�tāąčĒĖččā�u�t�t�t�t����������� ����)�*�/�0�)��������:�7�.�-�.�:�G�S�^�Z�S�G�:�:�:�:�:�:�:�:�`�Z�S�G�<�;�D�S�l���ݽ�����齷���y�`�л��������������ûлܻ����������ܻн������������ĽĽĽ���������������������¿º­¨©²¿����������������������¿ĦģĞĥĦĳľĿ������������������ĿĳĦ����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��$�"�����$�0�=�G�I�E�=�;�0�$�$�$�$�$D�D�D�D�D�D�EE
EEEEED�D�D�D�D�D�D�E*EE#E*E7E@EPE\EiEuEyEzEuEiE\EPECE7E-E*FFFFF#F$F1F5F1F1F'F$FFFFFFFF�Ľ����½ĽнҽٽٽнĽĽĽĽĽĽĽĽĽ� 9 : 8 M X D - \ ' w ; = c 8 M T V m 8 U ^ - ) g K B - < 9 i Y o H O J i X l Y m ; # O h V ~ + X , B X = T R D F i  % $ P M a S , � Z P  w  /    k  �  q  �  ,  �  .  l  �  �  �  �  >  �  p  ]  �  �  �    \  X  �  �  �  '  *  �  �    �    �  o  p  �  ]  �  o  �  �  �  �  �  �  j  )    �  D  �  �  i  �  �  b      �  �  H  �  X  e  u��`B��o��t��o�t����罃o�����w��j��w�\)��o��`B�0 ż��ͽ@����
�q��������1�D���e`B���8Q콃o�'0 ż�h��h�C��u�\)�<j�q���,1�t��t���{����D����w�8Q�0 ŽY��y�#�D����o�H�9����e`B�T���u���P�u��9X���罃o���罛�㽓t�������9X�Ƨ�\��^5�I�^B%�B�B�B,�BA�By%B�B!��B��B5�B �&B�B��Br BD�B%A�4aB�B �jB��B"J�B��B�3B*jB�B�	BJLB�B*sB��BW#B��B��BXB&�B*1B5%.B��B-�B�B��BK
B ��B��B��Bf�B!E�B!r�B
�B��B]�B)YB.q�B
B	W�B0�Bl B$��B%�nB
1�B
4cB�BքB��B#�B��B�ZB�tB@B��B��BL�B��B~�B��B" )BʶB=�B �JB��BJ�B-�B��B5�A�p<B?�A���B�B"9DB��B|rB>B��B		,BDRBAmB5B�=BO[B �B@�B��B&��B)��B4�cB�B,�OBL�B��B@�B ��B��BUxBE�B!?OB!O*B�B��B|B) )B.@)B?�B	�B*�Bw�B$��B%��B
>�B
-�B��BBE�B>�B�_B�sB�A59=A��:@��@?�A��?@�aBAsZ�A�O�A��dA�q�@���A�9�A�4�A��A�X�A�PA�@BC���A���A�MaACB}�A���A�a�A��kAV�;AXZ�Ay��AB��A�~�Ao�f@�r�?�R�A��9A��A��AM/�A:>�@��FAʔ�@vŀ>�a	B�_A>�A��AX�]?�&@8!;A�يAL82A��@��eAk�mA�WJA��2A��A�a@��hA#��A�~A��A�rC�'�B
C�K	C���C���A(A3
A���@���@B�A��@�a�As�A���A�A��f@�J�A�,:A�{�A���A��{A��A�meC���A��A��AB�B	4zA��tA��-A���AX�\AWr�Az�AC#A�idAq��@��?��4A�t A�C�A��+AL�A:�A	A�y_@y�>��B�mA=`A���AY$�?���@4�iA�;"AL�A�4\@�eAl�oA݀A�n�A�ARc@�(A#?_A�h�A��A�J;C�(�B
?IC�H]C��kC���A(�   ?            !   O   ;                               !      *   	         $   
      (                                        6   1            
                     '         	         $                              
                     7   5            #   3               1      -            #         '                  #         7            5   =                           !      %                  5                                                   5                  1                     '                                                7            1   =                                 %                  #                                 No�O	A�O!�lN8��N ��PP"�OY6N�aN�XjN��;O�X�PBPNZ�IO8��OO��N�TDO��aN^�P#�uN���Nf��O��O�AN��mOa|9O��ZO&4�OL��NƊMޭ�O*B\O��NN�eOMٝPg��O.�N=NWW�P.�sP��hN�y�NV��N�UuNA��N�0�N�`�OY�*N��Ov�N�V O���N�}�N�N��%O6�NW�O�fOWYNM�O�O LN2�JN��9N���N�V=NйN:ŚNE�f  �    �  �  �  L    W  �  �  �  	  �  �  X  �  �  �  b  -  �  �  �  `  �  �  �  V  �  �    =  �  �  
  a  J  �  r  �    �    �  F  �      �  _  7  �  `  �  �  C  �  Q        �  �  �     u  �  �=C�<t�:�o;D���ě���C���P�ě����ͼ#�
�D���e`B�49X�D����1�u��/�u�ě���t���t����
��/��j������P������`B��/��/��`B����h��h�������o�\)�\)�C��\)�\)��P��P�49X�0 Ž0 Ž@��49X�8Q�H�9�H�9�T���Y��Y���%����u��\)��7L��7L��O߽��㽛�㽛�㽧?|�	'&(										��������������������z�������������zxssuz� !���������������������������������6BC)������>BN[fgtuzzutg[NB<85>��������������������#./79//# ��
#%-/50#
�������������������������",,(,)��������569BO[aa[OB655555555�����������������������������	 ������������


	��������OTmz�����{xtma\TPOMO��������������������Ta������������zaVRQT�������������������������������������������������� ���������������� �����������

!##$#
��������������������6;BN[bgt����tg[NE>56��������������������:EFHTUadntz�~ynaUH4:269BDOOOGB<622222222��������������������46BOQU[^[VOD;6126:<4?BO[ht�����t[OCA=<<?T[ehitw}th[UTTTTTTTTR^ht��������}toh[QR	#7Db{����sf<0#
	�����������������~�������������������������������������������������
#!���������5BNUWO7'�������������������������KOO[[\hnlh[OKKKKKKKK��������������������GHUaemca`UMHGGGGGGGG��������������������;<>GHSUY\\ZUH<;;;;;;����������������������������������������sw~�������������}tssxz�����������zvwwxx��������������������z{���������{vsstzzzz
� ())59ABHB5))!N[gtz������tgd\[UPLN���������������������
)/1.)�������
#'.010-%#
��,0<<DD=<<0%$,,,,,,,,fgjtx����������trigfaght�����������{kgba��������������������#'/3752/#"!�����

�������������

�������������

  ��-/<CHIOH<;/.--------.5BNOPNB5*..........�������(�2�/�(�����������������#�/�7�7�<�@�?�<�/�#�����»��������ûϻлܻ�������������ܻлºֺѺɺȺɺкֺ����ۺֺֺֺֺֺֺֺ��n�l�h�f�n�w�zÇÇÇ�{�z�n�n�n�n�n�n�n�n�'�������'�M�V�T�\���������r�Y�4�'�����~��������������������ÿ������������H�G�@�G�U�U�a�b�n�z�{�z�z�z�z�n�a�[�U�H�<�8�<�<�H�U�a�a�e�a�U�H�<�<�<�<�<�<�<�<²¯±²³º¿����������������������¿²�Y�@�7�4�@�M�Y�f�k���������������r�f�Y�2��������5�N�s�����������|���{�N�A�2�H�F�;�2�5�6�;�H�K�M�Q�I�H�H�H�H�H�H�H�H�B�6�2�/�)�)�)�6�B�O�Z�[�b�e�l�h�[�O�E�BàÝÓÐËÇ��y�|ÇÓàìòü��ýùìà�����������������������������������������5�+���(�5�A�N�Z�g�������������g�Z�A�5FFFFF$F$F0F1F1F1F&F$FFFFFFFF��ĺĺĶĹĿ��������!�.�"�������������������������������������������������޾f�d�_�Z�W�Z�f�s�t�����������x�s�l�f�f���������������������-�5�9�9�6�0�$��/�%���� �"�/�;�O�T�a�m�u�t�i�T�H�;�/��	�	���	���"�/�0�;�=�;�;�/�"����œŇłŇňœŠŭ��������������ſŹŭŠœ�׾ʾ¾��žʾҾپ���	������	����׾����پ־׾޾������	������
�	���Ŀ��������������������Ŀѿݿ���ݿѿľs�h�f�d�f�q�s�u�����z�s�s�s�s�s�s�s�s��������	�����	���������������������m�g�V�]�`�m�u�����������������������y�m�_�X�P�W�h�v�}���������������������x�l�_�e�^�Y�V�Y�d�e�r�t�y�y�r�e�e�e�e�e�e�e�e���������������������������������������������t�Z�A�2�)�D�Z�s����������������������ݿҿɿƿ��Ŀѿݿ���	����$����侥���������������������������������������A�?�4�-�(�#�(�4�A�C�M�O�T�M�A�A�A�A�A�A������s�v�������ּ����%�)�'�����ּ���ä�v�Z�<�%��&�<�aÇù��������	�����Ż!���!�!�-�:�F�H�M�F�:�1�-�!�!�!�!�!�!�Ϲùù��������ùϹڹ۹ֹϹϹϹϹϹϹϹ�ƧƞƚƐƒƚƧƲƳ��������ƳƧƧƧƧƧƧ�M�J�B�K�M�Z�f�g�f�a�Z�R�M�M�M�M�M�M�M�M�����������������������������������������	��������ݾ����	���
�	�	�	�	�	�	�e�^�Y�V�T�S�X�e�r�~���������������~�r�e�����������ɺֺٺ�����ֺ˺ɺ�������ŹŭŠŞŕŔŎŔŠŹ������������������Ź���������������������ʾξоʾþ���������� ����� ���(�A�g�s�������g�Z�N�5��� ��
�
���'�*�4�@�B�K�G�@�4�'������`�]�`�f�m�y�y�������y�m�`�`�`�`�`�`�`�`�t�t�h�h�h�o�tāąčĒĖččā�u�t�t�t�t����������� ����)�*�/�0�)��������:�7�.�-�.�:�G�S�^�Z�S�G�:�:�:�:�:�:�:�:�l�S�I�J�W�l�y�����нֽݽ�ݽҽ��������l�л̻û����������û̻лӻܻ������ܻн������������ĽĽĽ�������������������������¿µ²®²´¿����������������������ĳĦĦġģĦħĳĽĿ����������������Ŀĳ����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��$�"�����$�0�=�G�I�E�=�;�0�$�$�$�$�$D�D�D�D�D�D�EE
EEEEED�D�D�D�D�D�D�E*EE#E*E7E@EPE\EiEuEyEzEuEiE\EPECE7E-E*FFFFF#F$F1F5F1F1F'F$FFFFFFFF�Ľ����½ĽнҽٽٽнĽĽĽĽĽĽĽĽĽ� 9 : . M b S & \  ~ ? 8 c 8 T S \ m = U ^ +  G 7 D , : 9 i Y I H O J i X l S m ; , O Q V L + X  B X = T R D F U  % ' 4 / a S , � Z P  w  /  \  k  i  q  �  ,  �    7  ]  �  �  �    �  p  �  �  �  �  r  �  �  }  n  �  '  *  �  \    �    �  o  p  }  H  �  o  �  r  �  �  �  �  �  )    �  D  �  �  i    4  b  X  f  D  �  H  �  X  e  u  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  �  	T  	�  
�  
�  
�  �  �  b  �  �  x  (  �  B  
�  
E  	�  	.  �    �  �  �  �  s  R  3      �  �  �  �  �  �  �  b  .  �  �  �  �  �  �  �  �  �  �  �  �  o  F    �  �  B  �  �  m  �  �  �  �  �  �  �  �  �  �  �  v  p  m  i  f  �  �  �  �  \  �  �    :  c  �  �  �  �  �  �  �  R    �  �  ?  �  �  '  G  H  L  ?     �  �        �  �  �  S  �  O  y  �  �    W  t  �  �  7  �  �  �        �  �  �  E  �  #  6  �  W  H  :  ,    �  �  �  �  �  }  [  C  H  T  X  K  E  H  V  �    D  r  �  �  "  M  q  �  �  �  �  `  2  �  �  i  �    �  �  �  �  �  c  8    �  �  �  �  �  �  p  B    �  �  r  �  �  �  �  f  C  4  p  |  �  �  |  d  I  +    �  �  {  �    �      �  �  �  �  �  x  O  7  "  	  �  �  �  �    3  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  r  i  �    n  \  I  3      �  �  �  s  H    �  �  �  �  �  �  �  �    6  L  X  T  F  /    �  �  �  H    �  p    �  =  �  �  �  �  �  �  l  U  <  !    �  �  �  ~  A    �  �  Q  ]  j  f  Z  }  �  �  �  �  �  �  m  E    �  �  O  �  3  {  �  �  �  k  U  =  $    �  �  �  �  g  M  2    �  �  �  j  ?  T  `  a  Y  E  %  �  �  �  X    �  x  2  �  �  "  �  �  -  (  $  #  $  %  %  &  %  $  !          �  �  �  �  �  �  |  v  q  l  f  a  Z  S  L  D  =  6  -  "       �   �   �  �  �  �  �  �  �  �  �  �  �  y  S    �  �  U  �  �  
  }  V  q  �  �  �  �  y  h  [  E  &    �  �  i    �  �  T  �  )  $    :  Y  U  G  4      �  �  �  �  k  C    �  �  y  z  �  �  �  �  �  �  z  o  U  0    �  �  �  �  c  >  �  �  �  (  M  m  �  �  �  �  �  s  Y  4    �  }  +  �  I  {    �  �  �  �  �  �  �  �  o  X  ?  #     �  �  R    �  �  �  R  U  O  F  >  6  .  %      �  �  �  �  �  Z  &  �  �  q  �  }  w  p  j  c  ]  W  P  J  A  7  ,  "         �   �   �  �  �  �  �  �    ~  |  {  z  x  u  r  o  l  i  f  c  `  ]    �  �  �  �  �  �  �  �  n  `  X  P  D  2        �   �   �  �  9  2  2  2     	  �  �  �  �  H    �  ~  N    �  �    �  �  �  t  _  I  3      �  �  �  �  �  �  �  �  �      �  �  �  �  �  �  �  �  �  j  H  $  �  �  �  �  c  ?    �  
  �  �  �  g  <    �  �  �  t  P  #  �  �  �  �  c  6   �  a  W  R  S  A  +    �  �  �  �  �  {  W  ,  �  �  �  =   �  J  M  P  S  T  O  J  E  9       �  �  �  �  `  :     �   �  �  �  �  �  �  w  g  W  H  8  7  C  P  ]  i  X  =  "    �  [  q  f  W  >    �  �  �  v  @     �  ~  3  �  |  �  -    �  �  j  N  3    �  �  �  �  T    �  7  �  �  �  <  m  6      �  �  �  �  �  �  �  �  �  �  �  �  �  v  f  V  G  7  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  `  L  7          �  �  �  �  �  �  �  �  �  �  �  �  �  i  R  :  "  �  �  �  �  �  �  �  s  g  U  >  "  �  �  �  �  b  7    �  F  B  ?  <  4  +  "          "  .  >  V  n  �  �  �  �  Z  V  j  �  �  �  �  �  �  s  [  B  )    �  �  �  �    `    �  �  �  �  �  �  t  S  2    �  �  �  �  _  @    �  �        �  �  �  �  �  �  �  �  �  �  �  �  p  g  a  \  W  �  �  �  �  �  �  �  �  �  �  u  `  G  (    �  �  F  �  �  _  V  N  E  =  3  )          �  �  �  �  �  �  �  �  �  7      �  �  �  �  r  �  �  �  �  �  M  �  �    �  �  U  �  �  �  �  �  �  x  l  `  R  D  6  &      �  �  �  �  �  `  ]  Z  W  T  Q  N  F  =  3  )        	       �   �   �  �  �  �  �  �  �  |  p  _  N  9       �  �  �  1  �  8   �  �  �  �    w  m  ^  L  9  "    �  �  i    �  a  �  m  �  C  4  %      �  �  �  �  �  �  �  p  \  G  1    �  I  �  �  �  �  �  �  �  �  �  �  �  �  �    A  �  �  3  �  U  �  (  8  E  N  P  L  @  /      �  �  �  L    �  r  
  �   �      �  �  �  �  �  �    j  X  H  7  '     �   �   �   �   j  �  �  �                �  �  �  r  +  �  �  I  �  �  e  o  w  }  v  d  H  *  	  �  �  �  e  0  �  �  �  c   �     E  N  X  b  p  ~  }  p  d  P  <  (      �  �  �  �  �  x  �  �  �  �  x  n  d  X  L  ?  0       �  �  �  �  �  �  �  �  �  �  �  k  V  A  ?  E  (    �  �  x  @    �  �  2   �     �  �  �  a     �  �  =  �  �  =  �  u  	  �    �    y  u  M    �  �  @  �  �  @  �  �  x  A    �  �  �  �  Q  %  �  �  �  �      �  �  �  �  �  �  J    �  �  u  D    �  �  m  U  ;    �  �  �  �  �  j  M  1      �  �  �  "  q