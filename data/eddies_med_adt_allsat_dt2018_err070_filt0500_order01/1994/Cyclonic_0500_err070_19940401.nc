CDF       
      obs    Q   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?����l�     D  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Mš.   max       P� �     D  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��+   max       <���     D   4   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��R   max       @F��Q�     �  !x   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @vi�Q�     �  .    effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @Q`           �  :�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�V@         D  ;l   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��G�   max       <e`B     D  <�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B4�E     D  =�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��L   max       B4�7     D  ?8   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >���   max       C�.V     D  @|   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�I�   max       C�(�     D  A�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          K     D  C   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =     D  DH   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9     D  E�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Mš.   max       P�qd     D  F�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���$tS�   max       ?�y=�b�     D  H   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��+   max       <���     D  IX   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��R   max       @F���
=q     �  J�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @vi�Q�     �  WD   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @P�           �  c�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�e�         D  d�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?!   max         ?!     D  e�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?�w�kP��     P  g                        
                              2                              1   8   
         
   -                              	      
                   9      2         J   !            
               &                                    
      N]�gO�3N��iO�ZN�0uN��JN9FO�IN�h�Oݥ NړNR�9N=tvNT��N �O��
O�P:dzN��eO��_N<O�17N�-�N�eN���Nt|Og��P+�O�J�N�M�O��OHxdN��|P� �P*�OC�nN�PNv�=O��Ny�FN���O9�OeN-xO8��O	�LNG�N��O;�hOX=2N1�@PJ~.O2�Oِ�N��O�ڭO�tQO���O��O�.�Nѱ�N�3�N��Nn$N&��N�,zP12�O���OY�O��sN!�zO��/N��POZ��O��Nn'�Mš.NP8�O:��N-�1M�z<���<e`B<#�
<o<o;��
;o;o$�  $�  �o�o���
�o�o�#�
�49X�49X�D���D���T���T���e`B��C���C���C���C���t���t���t����
���
���
���
��1��9X��j�ě��ě�����������/��/��/��/��/��`B��`B��`B�����+�C��C��t��t��t���P��P��P����w��w�#�
�''''0 ŽD���H�9�H�9�L�ͽL�ͽP�`�P�`�m�h�u��%��%��+%/;?G@;/-#%%%%%%%%%%0<IOUbflknnb^UIE=<10 ))))!
����������������������
!
���������������������������������������������������������������������mnz�����������zsnnmm��������������������-/5;HMQTYTOH;/*)---- #./6<AC</$#        Y[`htwxtmh^[YYYYYYYY�� ���������


/<HU_a^UTE</#z��������������~trzELR[����������tNB98E�(����������
#/27>?<4,
����-5<BNRQNB5----------*6CIOUW_ejh\O6�������
 #
�������W[cgjt}����ztng`[UWW��������������������������������������@BO[t��������|t^[OB@}�����������������z}z��������������}wttzX[\hst��������tlh[XXMT]mz���������za\PLM��������������������bgnty�������tkg_bbbb#>I�������{U<0^het������������|[W^����������������������������������������^akmpvsnmaVW^^^^^^^^����������������������������������BBOQZ[_hjrh[OMB@=<BB������������������������������ ���������)4686)4BEEHOR[hsplmlh[O614������������������������������������������������������������Y^ehtw����xth[VRTTVYEHUadnuy|~��znaUHDCE��������������������������" �����������������������������������+5BHN[gog][QNB5/++++sw�������������ytors����(385'���������������������������)6O`knh[B6) ��������������������������������������S[bgtx|{tggg[TSSSSSS#$-03410#����

�����������ABMNPZUQNFDBAAAAAAAA./<HUZ^ZUH<1/.......�����'(�������+1/11*#�������*1)�������LV\g����������tg[NLLit������utiiiiiiiiii��
#0<MI<86-#
����
!#&$$#
GKRUanz�������znaTJGotu�������������ttno#0<>><0'#��������������������8<HHNMJH<:9888888888egt{�����������smgde��������������������./8<=<40/.+)........���������������������������������������弽���������������ʼּּ�����ּռʼ��ֺͺɺ��ƺɺֺݺ���������ߺֺֺֺ��n�d�a�V�U�J�U�a�n�zÇÓÔÝÜÓÇ�z�n�n�����������������������������������������a�Y�U�T�U�V�X�a�i�n�t�y�x�n�a�a�a�a�a�aǈǄǈǏǔǡǫǬǡǔǈǈǈǈǈǈǈǈǈǈ���������������������������ʾԾӾоʾ������������������������������������������Ҿ׾ʾ�����`�b������վؾ���� ������������������������������������������������H�@�<�9�<�=�H�U�Y�]�U�S�H�H�H�H�H�H�H�H�H�C�;�2�2�;�H�N�T�W�T�O�H�H�H�H�H�H�H�H�����������������������������������������f�^�Z�Z�Z�f�s�����x�s�f�f�f�f�f�f�f�f����!�"�%�"�)�6�B�L�O�h�p�n�W�O�6�)����������!�9�T�a�y���|�z�m�a�H�;�"���Ŀ��������������Ŀѿݿ���
��
�����ľ�����~��������������������������������g�[�5�#��������5�B�[�t¥�t�g�����������������������������������������.�	��������	��.�G�T�a�^�W�P�L�S�G�.���������������������������������������������������������������	������	�������������������������ĿſɿοǿĿ������������������������������������������������m�e�c�e�c�_�`�m���������������������|�m�������ǺҺܺ���!�F�f�s�|�x�l�S�:��ֺ����������������ûлܻ����� ����ܻлû��r�f�e�a�e�f�g�h�o�r�s�~����~�~�w�r�r���������5�N�g�s�������p�g�A�(��6�1�)�)�)�6�>�B�F�O�[�]�h�m�m�h�[�O�B�6��������������	�	������	�������������p�N�C���g�������������������������m�T�;�3�C�A�6�;�a�z�������������������m�Y�W�N�L�G�@�?�L�`�e�r�~���������r�e�]�Y�����ܹչܹ߹������	�
�������������ƳưƧƥƧƳ������������ƳƳƳƳƳƳƳƳàÓËÇÂ�x�r�|ÁÇÓàæìùþ��ùìà�����������������������������������������_�_�\�_�l�p�x�������������������x�l�_�_ŔŌňŋŒŔŠŭŹź����������ŹŭŠŔŔ�T�G�;�.�"��	�����	��.�;�G�T�]�b�`�^�Tìäâìóù����ùùìììììììììì���ܹϹù��������ùϹܹ��������������y�p�m�`�`�\�`�m�y���������������������y�����������
��#�0�4�0�#���
����������������������������$�%�&�$������������~�{�����������Ľͽ߽�ݽнĽ����������������ùϹܹ����������ݹϹù������'�3�=�3�+�'�����������Y�O�`�z��������&�)����ּ������r�Y�s�g�\�Z�Z�c�g�s�t���������������������s��ʾ��������ʾ׾�����	�����������������������������������������������ŹŭŜ�ŇŠŭ������������������������Ź�@�4� ������4�@�M�Y�f�n�n�k�e�Y�M�@�s�g�U�N�E�P�Z�g�s���������������������s�����y�v�y���������������������������������x�s�g�^�Y�[�l�w�x�}�~����������������D�D�D�D�D�D�D�EEEEEEED�D�D�D�D�D���������)�3�6�6�6�0�)����������������(�4�7�<�4�(�%������������������!�'�"�!���������������H�D�<�;�<�H�U�a�f�a�U�P�H�H�H�H�H�H�H�H���������	����	������������	�����!�.�S�l���������}�l�S�.����ݽнĽнؽ�����(�4�A�C�4�(�����ùìÑÏÕÓàçìù������������������¿¦¦¿����������������������¿�Y�T�Q�T�Y�f�h�j�f�e�Y�Y�Y�Y�Y�Y�Y�Y�Y�Y�л����������Ļ̻лڻ����������ܻ�E*E)E*E4E7ECEHEPE\EiEsEiEeE^E\EPECE7E*E*�Ŀ����������������Ŀѿѿѿֿ���ݿѿ��/�$�#�����#�&�/�<�H�U�`�\�S�H�H�<�/�����������������������������������������������������������������������D�D~D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ĿĹīĦġĦĬĳĽĿ������������������Ŀ����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� . 5  R * U T - # m ( 1 4 R R Q M 1 3 ` ` R ^ J 4 K ~ ` 4 � P & 6 M . 8 P [ Z R U  d Y T : � A W Z I d I / x 0 , " V g 2 / H : K " D K � " w 8 X T 2 i X ^ ) Y n    o  C  �  S  �  �  m  5    }  �  j  V  t  5  u  J  0  �  �  }    <  �  �  �  s  �  �  '  M  �  �  �  �  �  �  s  .  �    �    p  �  9  �  �  �    K  2  �  �  #  �  �  y  �  �    �  �  �  I    O  6  [  �  �  �  
    T  �    z  �  ^  4<e`B;D��$�  ��C�;o��o��o�o�49X�����`B�#�
��`B�T���D���\)��w�u��t���w��o��󶼛�㼬1��j�������7L�����`B�8Q�#�
����o���C�������/�D���+��w�0 Ž49X�t��D���t����C��0 Ž}�\)��E��49X��1�#�
�]/��G���\)��7L�}�}�H�9�Y��0 Ž49X�T�����w�e`B�e`B��t��u��\)���P������C��e`B��o��7L��t���C����A��B'tBt�B9�B$DBr]B�B4�EB�B ��A��hBM?B�BZ�B*;BeSBgB	�HB��B:B��B0IBa�B	�B�BF;B"AB�|B~�Bq�A��B~VB	զB'�B
�SB">7B�gA�"`B��B��B�UB�B��B�BA�B*��Bp7B�HBѨB�B�B-	wB81B�B�B�B��Bv�B��B �B�]B	B�B%DB#�"B?"B�B:SB�xB"B
 B��B$}�B�B̆B
�lB%m�B@6B� B
Q�BU�BL#A��LB&�MB�NB@�B$?JB>B�@B4�7B��B!��A��vB@�B<2B�OB7SBYDB�B	ŞBRB�B¸B0*�B|B	�[BFYBsB*_B@BC�B=$A�	6B��B	±B&�B
�B"<�B2�A��ZB��BәB��B��B,�B1uB@CB*�7B��B )B��B�7B?vB-/�B;�B??B��B=\B@�B@B?�B ��B��B	CB%>�B#�4B>!BYB��BD�B�4B
=IB�B$@BB?�B��B
��B%?�B?&B�?B
F�B��BA�A�B�@�5<@?X	AȥI@�A���B`�AM�A�"�AM+ZA�1xA��`A�-�A���ABA�\A�Ay�4AHkhA��AHP�A`�A�|wA��CAw�A���Am�b@`�7@���?�qA��dA�LAZ"\A��`A��%?��=?'^@B��A�^A��@���A�TAa�IA��<>���An�vA��B�A#��>���?���@��YA� �AU��A�5�A��@�5A�n�A�ϼ@�KC�BA���A5�8@`�A�8AX��A�A2ьAΌ�A��@�-s@���C���Ay?�A�sA# �B��C���A�`}A�}}C�.VA�y�@�;@>��AȔB@��A��BHFAM�$A�iAKdA�v�A�*[A�pA�PAA��A�	oA���Ax��AG��A��5AI
&A]��A�nqA�kAwQUA���Am�@T6@���?��pA�޾A��AZ�bA��5A�w�?�L�?-6�B�OA��A���@�s3A��eA^�.A�~o>�I�Am؏A�GB	,�A"��>���?���A9eA���AUA�g�A��7@�5A�v�A��_@�-�C�?�A�
/A6s�@`�lA��bAY =A �A1�A�w�A�oy@��7@���C���Ay(AüTA"B�+C���A�o3A� C�(�                                    	                  2                              2   8               -                              
      
            !      9      3         K   "                           &                                    
                                    )                     '   +      +      '                  1   !      #         =   /                                                   9      !         !         !                     -      !   #      %                                                         '                              %                        +                  9   !                                                   9                                             +         #                                 N]�gNɉ%N��iN��'N�0uN�ON9FN���N���O���NړNR�9N=tvNT��N �O)\�O��O�SN��eO�>�N<O�~HN�-�N�eN���Nt|N~�aPS�O���N�M�O!^�O:��N�+FP�qdO��DO)�N�PNv�=OXd4Ny�FN���N��0OeN-xN.�'O	�LNG�N��@OD�N��N1�@PJ~.OF�O[@HN��O�x Of�O�&�O�AOh��N�eN�3�N���Nn$N&��NÒgP+$�OOkQO!�,O��sN!�zOO=	N��<OH�}N��Nn'�Mš.NP8�O:��N-�1M�z  d  �  �  $  %     '    �  �  �  �  L  �  �  D  T  �  �  �  �  �    �  z  p  v  �  �  �    �  �  j  �    �    �    �  R  �    �  p  �  �    F  9    �    *  4  
    Q  �  �  @  �    <  z  �    �  (  �  s  �    �  l  V    �  b  J<���<49X<#�
:�o<o�o;o�o�D���o�o�o���
�o�o��t������/�D���u�T����o�e`B��C���C���C���j��1��j��t��+��1��1��9X������j��j�ě���/������������/��/����/��`B��h����P���+�\)�T���t����ixս'�w�#�
��w��w�#�
�#�
�',1�,1�0 Ž8Q�D���H�9�]/�T���P�`�e`B�P�`�m�h�u��%��%��+%/;?G@;/-#%%%%%%%%%%7<ITU\bihebUIB<67777 ))))!
����������������������
!
���������������������������������������������������������������������sz�������zzsssssssss��������������������-/5;HMQTYTOH;/*)---- #./6<AC</$#        Y[`htwxtmh^[YYYYYYYY�� ���������

#<AHSRN=<9/+#"��������������������W[gt���������tjXQPRW�(�����������	 #/5<<:52*
��-5<BNRQNB5----------!*6CMQUXZXOC6*������
 #
�������W[cgjt}����ztng`[UWW��������������������������������������}���������zu}}}}}}}}�����������������||�w|��������������yvwX[\hst��������tlh[XX^aemz~������zmmda_]^��������������������egptw����{togaeeeeee#I{�������{U<0emt������������ta_ae����������������������������������������^akmpvsnmaVW^^^^^^^^�����������������������������������BBOQZ[_hjrh[OMB@=<BB������������������������������ ���������)4686)NO[hkih[[ZOMNNNNNNNN������������������������������������������������������������Z[`ght~���|vtnh[VVWZFHOUZaipsqnka\UPJHFF��������������������������" �������������������������������������������+5BHN[gog][QNB5/++++��������������~xvvz������##�������������������������)6O\hlh[B6)	�������������������������������������S[bgtx|{tggg[TSSSSSS#+02400#����

�����������ABMNPZUQNFDBAAAAAAAA//0<HUX\YUH<2///////����&'��������&)/,..-)����������������LV\g����������tg[NLLit������utiiiiiiiiii��
#*0210&
����	

 #%#"

				HLSUaknz������znaUKHst}������������{tsss#0<>><0'#��������������������8<HHNMJH<:9888888888egt{�����������smgde��������������������./8<=<40/.+)........���������������������������������������弽���������������ʼּټ��ּѼʼ��������ֺͺɺ��ƺɺֺݺ���������ߺֺֺֺ��n�k�a�`�a�c�n�zÇÈÓØÖÓÇ�z�n�n�n�n�����������������������������������������n�k�a�a�^�[�a�g�n�r�s�o�n�n�n�n�n�n�n�nǈǄǈǏǔǡǫǬǡǔǈǈǈǈǈǈǈǈǈǈ�����������������ʾʾ̾ʾƾ����������������������������������������������������Ҿʾ�����l�e��������ξ׾���������׾������������������������������������������H�@�<�9�<�=�H�U�Y�]�U�S�H�H�H�H�H�H�H�H�H�C�;�2�2�;�H�N�T�W�T�O�H�H�H�H�H�H�H�H�����������������������������������������f�^�Z�Z�Z�f�s�����x�s�f�f�f�f�f�f�f�f�)�'�%�'�)�-�6�B�O�h�i�h�c�[�O�O�B�=�6�)�"������,�;�H�T�a�n�w�l�a�[�H�;�/�"�����������������ѿݿ����������ݿĿ���������~��������������������������������t�g�[�5�)���
���5�N�[�g�t�t������������������������������������������	�������	�"�.�;�G�M�L�J�H�F�I�G�;����������������������������������������������������������������	������	�������������������������ĿſɿοǿĿ������������������������������������������������m�j�i�m�t�y�����������y�m�m�m�m�m�m�m�m���������ɺֺߺ����F�_�i�j�T�:��ֺɺ��û��������������ûлܻ����������ܻлúr�f�e�a�e�f�g�h�o�r�s�~����~�~�w�r�r�5�.�(��!�(�5�<�A�N�Z�h�o�g�g�Z�R�N�A�5�6�2�*�*�*�6�B�O�[�[�f�h�l�k�h�[�O�B�6�6������������	������	�����������������r�E�"��(�Z�������������������������m�a�R�M�N�K�L�T�m�z�����������������z�m�Y�R�L�H�B�B�L�Y�e�r�u�~��������r�e�c�Y�����ܹչܹ߹������	�
�������������ƳưƧƥƧƳ������������ƳƳƳƳƳƳƳƳÓÍÇÅ�{�wÁÇÓàâìùüþúùìàÓ�����������������������������������������_�_�\�_�l�p�x�������������������x�l�_�_ŠŕŔŐőŔŚŠŭųŹ����������ŹŭŠŠ�T�G�;�.�"��	�����	��.�;�G�T�]�b�`�^�Tìäâìóù����ùùìììììììììì�ù¹����ùϹҹܹܹܹڹϹùùùùùùùÿy�p�m�`�`�\�`�m�y���������������������y�����������
��#�0�4�0�#���
���������������������#�$�%�$�����������������������������������������Ľɽнٽܽн������ù����������ùϹܹ����������ֹܹϹú����'�3�=�3�+�'�����������Y�O�`�z��������&�)����ּ������r�Y�s�o�g�^�\�e�g�s�z�������������������s�s�׾ԾʾȾ¾þʾ׾�����	�����������������������������������������������ŠőŃŋŠŭŹ��������������������ŹŭŠ�M�@�4�+�����'�4�@�M�Y�]�f�g�f�c�Y�M�s�g�X�R�P�W�Z�i�s���������������������s�����{�x�w�z�������������������������������x�u�j�a�^�a�l�t�x�{������������������D�D�D�D�D�D�D�EEEEED�D�D�D�D�D�D�D���������)�3�6�6�6�0�)�������������(�4�5�;�4�(�"��������������������!�'�"�!���������������H�D�<�;�<�H�U�a�f�a�U�P�H�H�H�H�H�H�H�H�����������	����	����������������
���	��!�.�S�l���������|�l�a�S�.������ݽ޽�������(�.�=�:�4�(�����ùñìÔÒÓàìùý������������������¿¦¦¿����������������������¿�Y�T�Q�T�Y�f�h�j�f�e�Y�Y�Y�Y�Y�Y�Y�Y�Y�Y�л������������ûλлܻ߻��������ܻ�E*E*E*E5E7ECEJEPE\E\EcE\E\EPECE7E*E*E*E*�Ŀ������������������Ŀοѿտ���ݿѿ��/�.�#��#�$�/�<�>�H�P�U�V�U�M�H�=�<�/�/�����������������������������������������������������������������������D�D~D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ĿĹīĦġĦĬĳĽĿ������������������Ŀ����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� . <  = * � T " & g ( 1 4 R R = C  3 T ` F ^ J 4 K 9 Y 5 � B ' 1 I & - P [ ] R U  d Y 1 : � 0 R 8 I d " $ x + &  W c - / E : K # > ; � " w & S G 3 i X ^ ) Y n    o  �  �  �  �  �  m  �  �    �  j  V  t  5  �  "  x  �    }  =  <  �  �  �  �  u  E  '  X  �  �  �  �  t  �  s  �  �        p  B  9  �  �  ]  %  K  2  C  �  #  U  �  *  �  K  �  �  �  �  I  �    �  �  �  �  �  �  �  �  �    z  �  ^  4  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  ?!  d  [  S  J  A  5  )          �  �        !  .  :  F  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  [  ,  �  �  l  �  �  �  �  �  �  �  �  n  X  >  !    �  �  �  ~  [  �  x  �  �  �  �    "  !    �  �  �  j  *  �  |  �    L  k  w  %             �  �  �  �  �  �  �  �  t  U  2     �   �  �  �  �    
      '  >  e  �  �  �  g  J  ,    �  �  �  '          �  �  �  �  �  �  �  �  i  )  �  �  ^     �  �  �                 	  �  �  �  �  �  x  Q  *     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  S  9    �  z  �  �  �  �  �  �  �  �  �  �  �  �  �  ^  5    �  �  Q   �  �  �  �  �  �  �  �  �  �  �  �  �  r  [  E  .     �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  X  7  L  H  D  ?  ;  7  3  /  +  '  %  %  %  %  %  %  %  %  &  &  �  �  �  �  �  �  �  p  \  G  2      �  �  �  �  �  j  N  �  �  �  �  �  ~  m  \  L  ;  (    �  �  �  �  �  �  �  �  �  �  �    0  ?  D  D  ?  2    �  �  {  1  �  m    �  |  �  �    :  M  S  O  B  0  9  0    �  �  �  s  4  �  c  �  "  P  �  �  �  �  �  �  �  �  �  �  �  �  j  '  �  Y  �  �  �  �  �  �  �  �  |  l  Z  I  4      �  �  �  �  �  �  �  g  �  �  �  �  z  a  C     �  �  �  z  7  �  �  p  P    �  �  �  �  �  �  �  �  �  �  �  �  �  }  m  R  8       �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  d  D    �  �  |  6       �  �  �  �  �  �  �  �  �  �  �  �  z  i  I  %      �  �  �  �  �  �  �  �  {  s  k  a  V  K  @  4  %       �   �  z  o  e  Z  P  F  <  /  !      �  �  �  �  �  �  �  �  �  p  n  m  k  j  h  f  e  c  b  d  k  r  y  �  �  �  �  �  �    "  -  6  L  P  N  e  q  t  h  [  N  B  '    �  �  �  D  �  �  �  �  �  �  �  w  S  $  �  �  �  v  2  �  q  �  c  �  �  �  �  �  �  o  S  1    �  �  F  �  �    ]  �  �  W  v  �  {  o  U  ;    �  �  �  �  [  %  �  �  \    �  �  �  �  �  �  �                    �  �  �  M    �  o    �  �  �  �  �  �  �  �  }  i  M  6  '    �  �  |  =  �     �  �  �  �  �  �  f  G  (    �  �  �  h  6    �  �  k  7  [  g  M  '  �  �  �  h  6  �  �  {  3  �  �  z  8  �  q   �  �  �  �  y  �  �  �  �  �  g  N  (     �  �  �  �  �  �  g            �  �  �  �  �  �  �  c  E  #  �  �  �  �  q  �  �  �  �  �  �  �  �  �  �  �  �  z  x  �  �  �  �  �  �       �  �  �  �  �  �  �  �  �  x  i  \  U  N  G  @  8  1  �  �  �  �  �  �  f  A    �  �  v  7  �  �  �  h  0  �  �      �  �  �  �  �  �  �  �  �  �  u  e  P  3    �  �  �  �  �  �  �  �  �  �  w  c  R  H  9  %    �  v  �  w  �  [  )  3  >  G  N  R  R  H  2    �  �  �  �  e  .  �  �  e  D  �  �  �  �  }  f  J  +  
  �  �  �  �  �  �  W    �  M  �                      �  �  �  �  �  �  �  �  q  Z  f  f  c  Z  Q  J  =  '  9  �  �  �  �  ~  ]  :    �  �  �  p  n  l  i  e  ]  T  H  :  +      �  �  �  �  �  `  4  	  �  �  �  �  �  z  s  k  d  ]  U  L  D  ;  2  *  !        u  y  |  �  }  z  v  k  \  M  <  (    �  �  �  �  �  }  `  	          �  �  �  �  �  }  Z  0    �  �  B  �  �  �  �  �  )  ?  F  D  7     �  �  �  Y    �  b  �  �  �  2  �  9  )      �  �  �  �  �  �  �  �  �  v  j  ^  P  C  5  '    �  �  �  �  [  9    �  �  U    �    7  �  �  �  )  U  g  �  �  �  �  �  �  �  w  j  Z  G  .    �  �  �  v  D      U  �  �  �  �        �  �  w  *  �  Z  �  G  �  �  T  *  '  $  !        #  &  )  #      �  �  �  �  �  z  _  *  3  4  1  .  +  )  &  #         �  �  �  �  \    �  K  	H  	�  	�  	�  
  
  
  	�  	�  	�  	B  �  �  C  �  $  �  �  �  �  �          �  �  �  �  �  �  y  M    �  l  �  �  M    ?  N  I  9  &    �  �  �  �  ]    �  �  R    �    s   t  �  �  �  �  �  �  �  a  H  0  %  !       �  �  t  �  H  �  �  �  �  �  �  e  @    �  �  �  D  �  �  \    �  \  �  �  @  ?  =  9  4  -  "      �  �  �  �  �  ~  d  F  &  �  �  �  �  �    v  a  I  .  
  �  �  y  D    �  �  c  (  �  �         �  �  �  �  �  �  �  �  �  �  �  �  y  d  P  ;  &  <  3  *  !        �  �  �  �  �  �  �  �  �  �  �  �  �  r  v  z  y  w  r  k  c  V  H  7  "    �  �  �  c  0  �  p  �  �  �  �  k  Q  1  �  �  q  !  �  r    �  >  �  `  �  �  �  �  
    	    �  �  �  �  �  u  X  ?  -    �  
  :  h  �  �  �  �  �  �  �  a  =    �  �  �  �  �  �  �  n  �  D  (    �  �  �  �  �  ~  b  E  (    �  �  �  �  �  h  )  �  �  �  �  �  �  �  �  �  �  ~  d  I  -    �  �  �  �  �  �      C  h  o  s  l  _  N  ;  &    �  �  �  W    �  y    �    {  d  H     �  �  q  2  �  �  t  8  �  ~  �  �  �  �  �      �  �  �  �  �  �  �  �  �  r  >    �  z  -  �  u  ?  U  i  y  �  �  �  x  X  4    �  �  �  l  =    �  �  �  l  c  Z  P  H  B  <  6  *    �  �  �  �  �  �  w  a  K  5  V  S  Q  N  ;  %    �  �  �  �  �  w  ]  @  $    �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  f  O  6      �  �  �  �  �  �  �  b  Y  P  G  >  :  5  0  '    	  �  �  �  �  �  �  r  Z  B  J  4      �  �  �  �  �  v  `  K  ;  ,         �  �  �