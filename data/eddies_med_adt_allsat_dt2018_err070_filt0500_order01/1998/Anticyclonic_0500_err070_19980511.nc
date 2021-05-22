CDF       
      obs    =   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?ě��S��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�~v   max       P��$      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �L��   max       =�Q�      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�G�z�   max       @FFffffg     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�    max       @vq�Q�     	�  *   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @P@           |  3�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�k        max       @��          �  4   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �#�
   max       >�        �  5   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�	�   max       B0q�      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�q�   max       B0�      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?Jݖ   max       C�q�      �  7�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?aW   max       C�w�      �  8�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  9�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          M      �  :�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =      �  ;�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�~v   max       P��C      �  <�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�?|�hs   max       ?�|����?      �  =�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �H�9   max       >J      �  >�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @FFffffg     	�  ?�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�    max       @vq�Q�     	�  I   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P@           |  R�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�k        max       @���          �  S   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�      �  T   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?{�Q�`   max       ?�z����       T�               $   #   v                                 6   -         \   G   C      
   1               M   @         '      )                     
   !         O         &   7   4   	      �   	         �O: �N ؽN�iN��{O�A�O\��P��$M�~vN�NK��N�,�N/��O-ȥN�*�O�ߦO֝Or�!O�ӎO�}OX��O_�#PQ�P!��P�=�N�?%N<fP.ON���N��>O9��NWI-P�C�P�G'Ni�NlVO}��O��_O��aN|N�7OX{�N�JzNS�NӶ!N@�O���NFP~N���OǨ�N�
O&0vO~��P
��O��cNϊ�O�;fO۵(M�-�N�5N��oOָ6�L�ͽt��e`B�D���49X�ě����
�o:�o;D��;�o<o<o<t�<t�<49X<49X<D��<T��<e`B<e`B<�C�<�t�<���<�1<�9X<�j<ě�<���<���<���<���<���<�/<�/=C�=C�=C�=C�=\)=\)=��=�w=#�
=#�
=8Q�=@�=@�=L��=L��=]/=q��=y�#=��=��=�C�=�t�=��=��
=��=�Q���������������������y�����������yyyyyyyypmkjjtz������tpppppp����������������������������������������V[ft����������{tlg`V��	/LSX[URa�|o#�����������������������������������������9;BOYX[b[OCB99999999stw|�������������{ts��������������������������������������������������������.458BIN[gu{���g[NB5.��������������������������������������/;HNV[YTH;/"!#0<Un{~|nUI<0&��������������������������

��������5GLVWMB)���������

%01/%
����mjmz�����?PB����m��������������������������������������������
/HUZ_a`VH<"���������������������������")-)'������������� ����������������������������������)>ELNJB)������/BUgt������thaB��������������������!()6BINBB6+)!!!!!!!!$#*5BNX[_aa]QB5)$eqww{������������tge��������������������4-6BMORSQOGB<6444444��������������������WPU[h��������thfbecW+&(,/<GHQPHH</++++++
#'#"###
VXacmz|�����zmmaaaVV*./<HHMH</**********��������������������BBIO[dcc[OKBBBBBBBBB{u������������{{{{{{��������������������+6CO\g^\OLC6*%������ 	�������LINWdm�����������mTL���������������������	��������������� ���������������������������������������������������������������HC?<:/##/8<HHH���������

�����ĚĦĳĿ����������ĿĸĳĦĥěčċčĔĚ�O�[�[�h�m�j�h�_�[�O�M�K�O�O�O�O�O�O�O�O�a�n�zÇÓßÓÌÇ�z�n�a�a�]�a�a�a�a�a�a���	�
�����	������������������������������
���������ŹŭũũŧűŹ���������������������������������������������������5�g���������	�������s�S�(����ݿ꺤�������������������������������������������������������������������������������������z�x�u�z�{���������������������ûлۻܻݻܻлĻû»������������������������������������������������������Ҿf�s�|���������������������s�f�[�Z�`�f������������������������������������������������������������������������������������������������������������������4�@�M�L�4�(�!������ݻ������ �'�4�/�H�T�a�m�|�����m�T�;�/�"���
���"�/��������������������������f�F�K�X�a���ŭųŹſ����������ŹŭŦŠŔŒŏŒŔŧŭ�Z�f�����������������f�Z�M�4�,�4�:�A�Z���
�#�<�UŀŌŃ�l�U�I�#�
������������������������������������������b�_�c�m���s���������������i�X�A�(��������(�s�zÇÓàìóìàÙÓÇ�}�z�q�z�z�z�z�z�zE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eٿy����������`�G�;�"�������	�.�V�`�y�(�5�A�E�N�R�V�V�N�A�5�4�(�!��#�(�(�(�(�M�Z�f�m�l�k�i�f�_�Z�N�M�K�K�J�M�O�M�M�M�����	�������	��������������������6�C�O�W�\�a�\�\�O�O�C�8�6�5�6�6�6�6�6�6���������������ƚƍƎƊƁ�m�jƎƳ�����"�)�;�V�[�H�;�	�����o�j�s����������������¿²§²²¿��������������������f�r�t�v�v�r�f�Y�Y�V�Y�[�f�f�f�f�f�f�f�f�"�.�;�G�T�`�j�m�n�m�j�`�T�G�;�.�#���"�y�������Ŀοӿ̿��������y�l�w�v�y�r�p�y�Z�f�s�y�w�r�t������s�f�Z�R�A�.�(�(�4�Z���ʾϾѾʾȾ����������������������������{ǈǔǡǭǰǹǭǬǡǔǌǈ�{�v�t�{�{�{�{������������������������r�e�c�f�r��������������
�������������������������޼������� ���������������������������������������źŹŵŹŹ����������¥����(�4�A�M�c�s�w�v�f�Z�Q�A�4�(����<�<�H�J�M�H�<�/�*�-�/�;�<�<�<�<�<�<�<�<�/�<�C�H�L�I�H�<�9�/�,�%�#�,�/�/�/�/�/�/���лܻ���޻лû������l�b�l�x������������!�-�:�F�S�X�S�H�F�:�-�!������`�m�y���������y�i�`�T�S�V�T�Q�Q�R�T�Y�`�e�r�~�����������~�r�e�_�Y�E�>�>�@�L�U�ečĚĦĳĿ������������Ħā�t�s�ăāćč²¿���������(�)�%��
��������¹®¨²�B�O�[�h�t�v�}�t�r�h�[�O�J�C�B�A�B�B�B�B�������ý��Ľ����������y�i�a�\�l�y�������������ּ�����ּʼ�������������������!�(�(�!������������������������������z�z�����������6�*���������)�*�6�>�:�6�4�6�6�6D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DuDtDyD�D�  e Y 0 R 2 { v k L $ C 7 L 8 1 � 3 - @ p 5 D | S = p B g t O * T u S / J V O 8 ;  I Z : @ > L F M c ) R _ 5 % 8 C . 6 J    �  p  �  �  N  �  �  X  d  �    ^  s  �  C  X  �  �    �  M  �     W  �  4  �  (  P  �  �  9  #  �  �      L  �    �  �  n    h  y  k  �     8  �    �    �  :  �  #  ,    �#�
���o��`B<ě�<�h=�l�;o;ě�;ě�<�/<e`B<�9X<e`B<�<ě�<�h=��=m�h=o<�9X=�;d=�9X=� �=o=o=�t�='�=o=�w<�=��=�Q�=��=��=�t�=P�`=��P='�=@�=]/=ix�=0 �=e`B=L��=��w=P�`=}�>�=�+=�o=Ƨ�=�h==��=Ƨ�>^5?=��T=�{=Ƨ�>�  B�B
�nB	ӿBFB�B	�B;fB"�1B 7oB��B�<B��B `gBz�BĬByUB!m(A�	�B'<B;OB�Bi�Bp�B�B"=�B�B��B�B�
BF�B(B|�B	`MB�B�B�XBI
B0�BL�BCBMDB�0B$��A�B��B�uB�^Be�B��B�hB0q�B�B g�B�dB��B,�rB��B8,B�yB1�B�?B��B
��B	��BGsB:`B	ĭB@�B"��B -B�_B�}B�BB A�B��B��Bs�B!��A�q�B' �B\6B�PBA�BC�B?lB"4,B��B>�B�B�RB�mB>&BD>B	B�B�B�qB��B
�>BǾB@iB8UBJ�B�fB$�;A�}OBB��B�bB�<B�B��B0�B� A��B��B�B,��BV�B@B8�B��BťA�/�A�x1AȆ�A�C�A�A�o�A�R�@E�?JݖA�i�@�`,A��AD��Ar��A��A��U@�ӄA�@��A���ABE�A�[�A���A��A�T�C�q�Ab��A��A?�&A���B4Bm�A���A�Z�@ޮAd�}Aq�A>��ANh�BEk@�pA��A�A�j�A�ͺA:'�AÈeA���@��L@o��AjS�?지A�i�A�J�AڢA�/@�A�@hV�@�J�A� ]C���A��FAڂ�A�A�y�A�{�A�}A�r'@�	?aWA� z@�8QA��AD��As4�A�DA�@���A�s�@�l�A�WlAE�A��A�qA�egA� C�w�Ad�jA��|A>�A�m B=IBLA��A�@��*Ad�tAu��A=K�ANԕB4w@���Aѥ�A��A���A���A8�\AÁA�}x@�0�@s�Aj�?��OA���A�~	Aڃ�AO@��@dB�@�K A���C��               $   $   w                                 6   .      	   ]   G   D         2               N   A         '      )                        "         P         '   8   5   	      �   	         �                     M                              !      #         /   -   I         1               3   A            %                                    %            )   #         !            #                                                   !                     -         +               !   =            %                                    #            #                        O�wN ؽN�iN.��OSN�]O���M�~vN�NK��Nb�$N/��O-ȥN�*�O�ߦO֝Or�!O��O�gOO9C�O_�#O�R�O���P4$N�?%N<fO� N���N�KOO$�NWI-O�YP��CNi�NC�OKbDO��_Oq�0N|N�7O*�N}XnNS�N���N@�O���NFP~N�$O��yNܿ�O&0vO:�O�;kOe��Nϊ�O�;fOS�/M�-�N�5N��oO�b    2  9  {  N     	  �  �  �  �  �  E  N  r  �  �  	$  �  �  �  	Z  W  @  %  S  �  	  �  !  H  )  �  u    �  �  �  T    �  �  �  ~  >  y  $  q  �  =  �    �  �  ]  P  o  C  @  C  ��H�9�t��e`B�49X��o;ě�=y�#�o:�o;D��<49X<o<o<t�<t�<49X<49X<�1<���<�o<e`B=ix�=8Q�=H�9<�1<�9X=\)<ě�<���<�/<���=ix�<�<�/<�`B=�w=C�=��=C�=\)=��=49X=�w='�=#�
=8Q�=@�=D��=T��=P�`=]/=��=�o=��-=��=�C�>J=��=��
=��=����������������������y�����������yyyyyyyypmkjjtz������tpppppp����������������������������������������a^cgt}��������tkgaa#/<HOSUQHB</#����������������������������������������9;BOYX[b[OCB99999999wz��������wwwwwwwwww��������������������������������������������������������.458BIN[gu{���g[NB5.��������������������������������������"/;GQUWUOH;/",''0<Ibnsvvtm`UI<80,��������������������������

������)5=CEED=5)�������
 #"
������������&-/(������������������������������������������������
#/<JTYXUNH</#������������������������� ),)��������������������������������������������������)07:84$���4BXgt�������tnd^B"��������������������#))6BFJB@6-)########! "%).5BNX[\][YNB5)!eqww{������������tge��������������������4-6BMORSQOGB<6444444��������������������Z[^ht��������thghf\Z+-/7<?HNLHA<2/++++++
#'#"###
Yaaamz{�����zombbaYY*./<HHMH</**********��������������������BBIO[dcc[OKBBBBBBBBB}x������������}}}}}}����������������������+6CO\g^\OLC6*%��������������PKPXm������������mTP������
�������������	��������������� ������������������ �����������������������������������������������HC?<:/##/8<HHH���������
	�����ĦĳĿ��������ĿĵĳīĦğĚčČčďęĦ�O�[�[�h�m�j�h�_�[�O�M�K�O�O�O�O�O�O�O�O�a�n�zÇÓßÓÌÇ�z�n�a�a�]�a�a�a�a�a�a����	�����	����������������������������������������������������žŹŸŹ�����������������������������������������������(�5�A�`�g�]�N�A�5�(�������
����������������������������������������������������������������������������������������z�x�u�z�{�������������������ûллջлû����������������������������������������������������������������Ҿf�s�|���������������������s�f�[�Z�`�f������������������������������������������������������������������������������������������������������������������4�@�M�L�4�(�!������ݻ������ �'�4�;�H�T�a�l�u�x�m�T�H�;�/�"�����"�/�;�����������������������r�f�W�Z�b�f�r�ŠŭŹż������������ŹŭũŠŔŔőŔŕŠ�Z�f�����������������f�Z�M�4�,�4�:�A�Z�
��#�7�H�T�U�J�<�0�#��
�������������
�����������������������������y�z���������N�g�s�r�h�d�L�5�(���� ���$�(�0�A�N�zÇÓàìóìàÙÓÇ�}�z�q�z�z�z�z�z�zE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eٿm������y�i�T�;�"�	���������.�G�`�m�(�5�A�E�N�R�V�V�N�A�5�4�(�!��#�(�(�(�(�Z�f�k�k�j�h�f�]�Z�R�M�M�M�M�M�L�M�T�Z�Z�����������	����������������������6�C�O�W�\�a�\�\�O�O�C�8�6�5�6�6�6�6�6�6�������������� ����������ƳƧƣƠƠƦ���	�"�'�;�R�W�H�;�	�������u�o�s���������	��������¿²§²²¿��������������������f�r�r�s�t�r�f�]�Y�W�Y�]�f�f�f�f�f�f�f�f�.�;�G�T�Z�`�f�j�j�`�V�T�G�@�;�.�&�#�$�.�y�������Ŀοӿ̿��������y�l�w�v�y�r�p�y�Z�f�s�w�u�o�q�{�y�s�f�Z�U�A�1�+�,�4�M�Z���ʾϾѾʾȾ����������������������������{ǈǔǡǭǰǹǭǬǡǔǌǈ�{�v�t�{�{�{�{������������������������r�i�o�r��������������������������������������������������� ���������������������������������������żŹŶŹź����������¥����(�4�A�M�c�s�w�v�f�Z�Q�A�4�(����<�<�H�J�M�H�<�/�*�-�/�;�<�<�<�<�<�<�<�<�/�<�A�H�L�I�H�<�7�/�-�&�$�-�/�/�/�/�/�/���ûлܻ����ݻлû��������m�x������������!�-�:�F�P�G�F�:�-�!��������`�m�y���������y�i�`�T�S�V�T�Q�Q�R�T�Y�`�L�Y�e�r�~�������������~�r�e�d�Y�L�E�G�LčĚĦĳ��������������Ħčā�{�{ąĄĊč�����
���#���
�����������������������B�O�[�h�t�v�}�t�r�h�[�O�J�C�B�A�B�B�B�B�������ý��Ľ����������y�i�a�\�l�y�����������ʼּټ���ּʼ���������������������!�(�(�!������������������������������z�z�����������6�*���������)�*�6�>�:�6�4�6�6�6D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D}D{D�D�D�  e Y H B   " v k L + C 7 L 8 1 � . ! < p  * y S = h B h s O  S u P % J P O 8 6  I V : @ > F < F c ! O % 5 % & C . 6 E    [  p  �  Z  [    *  X  d  �  w  ^  s  �  C  X  �  O  2  �  M  `  (    �  4  �  (  �  �  �    �  �  t  �      �    w  z  n  �  h  y  k  �  �    �  �  �  �  �  :  �  #  ,    c  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�    	           �  �  �  �  �  �  �  n  [  L  N  T  U  U  2      �  �  �  �  u  T  4    �  �  �  �  c  =     �   �  9  /  #    	  �  �  �  �  �  �  �  �  u  Q  %  �  �  �  k  V  `  i  s  }  �  �  �  �  �  o  [  B  %  	  �  �  �  �  x  u  �  �  �    A  L  D  -    �  o  ,     �  ,  �  !  �  �  c  �  �  �  �  �     �  �  �  �  u  8  �  �  '  �  F  �    y  �  {  '  {  �  p  e  ~  �  �  �  �  �  N  �  �  '  �    �  �  �  �  �  �  �  y  d  O  =  .         �  �  �  �  �  �  �  �  �  �  �  �  �    n  ]  L  ;  (      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  i  Z  �  0  c  �  �  �  �  �  �  �  �  �  �  �  �  t  R    q  �  �  �  �  �  �  �  �  �  �  �  �  �  u  c  P  =  (    �  �  E  D  A  <  2  $      �  �  �  �  �  �  _  8  �  �  w  .  N  F  =  5  .  (  "      
  �  �  �  �  �  �  �  �  �  �  r  ]  K  >  0      �  �  �  S  '    �  �  z    �   �   D  �  �  �  �  �  �  �  �  �  �  �  {  `  @  !    �  �  �  }  �  �  �  �  �  �  �  �  �  �  ^  :      �  �  �  �  �  �  �  �  	  	#  	  	  �  �  i    �  n    �    �    l  �  �    +  F  b  z  �  �  q  ]  C  #  �  �  �  ^    �  >  �  ~  �  �  �  �  �  m  Z  F  0    �  �  �  �  �  S    �  9  ;  �  �  �  �  �  �    i  M  /    �  �  �  �  |  u  k  \  M  �  $  `  �  �  	  	9  	P  	Y  	W  	J  	)  �  �  E  �  �  �  �  �  2  �  �    '  =  J  W  T  K  ;  *    �  �    h  �  W  �  +  >  ?  =  7  <  9  3  5  1    �  k  �  j  !  �  !  "  �  %    �  �  �  �  �  �  e  B    �  �  �  �  m  L  /    �  S  >  *    �  �  �  �    S  !  �  �  �  Q    �  �  t  :     E  j  �  �  �  �  r  Q  %  �  �  [    �    R  �  
    	       �  �  �  �  �  �  �  w  T     �  �  B  �  �    c  �  �  �  �  �  �  �  r  _  C  '  	  �  �  �  v  Q  ,     �             �  �  �  �  �  x  _  Q  ^  N  8    �  �  �  H  A  :  3  -  %        �  �  �  �  �  �  �  �  �  �  �  �    j  �  �  �    &  )  !    �  �  a  �  p  �    #  :  �  �  �  �  �  �  j  A    �  �  �  O  �  �    @  �    �  u  [  C  0             �  �  �  �  �  �  �  |  j  G          #  )  .  #       �  �  �  �  �  u  b  P  <  )    B  x  �  ~  w  j  ]  N  >  +    �  �  .  �  h  �  �    F  �  �  �  �  �  �  p  W  :    
  �  �  �  �  j  C    �  c  [  �  �  v  X  3    �  0  �  �  �  L    �  L  �  #    �  T  L  D  ;  1  &      �  �  �  �  �  �  x  O  '  �  \   �        �  �  �  �  �  ~  e  I  +    �  �  �  �  o  5  �  �  �  �  �  �  �  �  �  �  j  F    �  �  b     �  �  `   �  c  n  w  ~  �  �  �  �  y  g  M  &  �  �  �  C  �  u  �  K  �  �  �  �  �  �  �  �  �  �    r  e  W  J  >  2  '      q  |  k  \  F  *  #    �  �  �  W     �  �  �  J  �  �    >  ;  7  3  .  '           �  �  �  �  �  �  x  d  R  ?  y  m  d  [  R  G  8        �  �  �  \  1    �  k  �  "  k  $         �  �  �  �  �  �  �  �  �  }  g  N  3     �   �  p  q  o  j  `  P  ?  &    �  �  �  c  7  
  �  �  �  �  �  5  �  �  w  V  )  
�  
�  
�  
=  	�  	�  	#  �  #  �  �  �  �  �  5  <  #    �  �  �  �  q  N  )    �  �  �  b  6    �    �  �  �  �  �  t  ]  M  9      �  �  �  �  �  �  c  4    �  �      �  �  �  �  �  �  �  �  �  g    v  �        p    i  2  �  �  |  _  P  \  ~  �  �  c    z  �  �  �  e  �  �    q  �  �  �  u  G    �  y    �      �  �  ^   �  ]  I  5  #      �  �  �  �  �  �  �  �  �  ~  m  [  H  4  P  @  '    �  �  �  �  �  �  �  �  �  �  c  0  �  �  y  7  	  �  d  �    @  b  o  b  '  �  M  �    O    �  
�  �  �  C  A  ?  I  Z  f  b  ^  P  @  /      �  �  �  �  �  �  v  @  9  2  *  "      	    �  �  �  �       #  "        C  5    �  �  �  �  i  E  #     �  �  �  `  /     �  �  �  �  0  l  �  �  f  7  �  �  p    �  &  j  O    �      �