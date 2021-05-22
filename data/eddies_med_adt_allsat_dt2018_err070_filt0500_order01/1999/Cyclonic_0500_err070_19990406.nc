CDF       
      obs    V   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�;dZ�     X  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�-�   max       P��     X     effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �\   max       <t�     X   \   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>������   max       @F1��R     p  !�   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ᙙ���    max       @vn�Q�     p  /$   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @Q�           �  <�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�         max       @�@         X  =@   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ����   max       ;�o     X  >�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��/   max       B0-T     X  ?�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~   max       B0@�     X  AH   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =z/�   max       C�T�     X  B�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >���   max       C���     X  C�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          =     X  EP   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A     X  F�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -     X  H    
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�-�   max       P Bv     X  IX   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��?��   max       ?�!-w1�     X  J�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �\   max       <t�     X  L   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>������   max       @F,�����     p  M`   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��G�z�    max       @vn�Q�     p  Z�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q�           �  h@   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�         max       @��         X  h�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         FH   max         FH     X  jD   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�|����?   max       ?�L�_�     �  k�      	                     	      =                                                   #         
      
      :      0                        "   %                  &                        	         "      %            0                  0          
                        	N��GN���O���Np0�N�jN�g�N�܋O�N���NNBP�lNS�&O8�NM�NdU�NDM�-�N��Oޯ�OõSN3|�O��#N;X�N�5N�Z�Nl�;N�"�O�,jO"�NZzN���O4� N��N�JP��O�O�kN>�2M���O�VlO.�Oc1N��OO�OуO�%N��gOd]N��O��O>��P#0NX³O0�MᤋN<�O;>1N��NPj N���O�S�N���N�>9NPDO��nN�/N�"�O.!O�{N�DFO�2lN�_�O��NK-�P�fN);�O��N��O���N���Nӷ�N��!N��O���N��Nl��<t�<t�;��
;D����o�D�����
�ě��t��t��49X�T���T����o��C���C���C���C����㼛�㼛�㼛�㼣�
���
��1��1��1��j��j�ě��ě����ͼ��ͼ��ͼ���������/��`B��`B��`B��`B��h���������o�o�o�o�+�C��\)�\)�t�����w��w�#�
�#�
�#�
�#�
�'''',1�,1�,1�,1�0 Ž0 Ž0 Ž0 Ž8Q�P�`�T���T���Y��Y��aG��aG��}󶽇+��O߽\nnpz���������ztnnnnn�� 
#'.#
	����������
#'+/,#
�������������������������������������������
###
������?BNO[]e[OB;7????????��#/HOQH;/#
����-/18;HKRRSPHB;0/----��������������������&-6Ohtvz���zh[B60&"&����������������������������������������0<HU^UQH<90000000000-/<GHUHG<<;6/)------}�������{}}}}}}}}}}��������������������%)36BNOPPOLHB:6)&!%%/H\amy�����znlU</&%/?HQUamz����zvaUH<89?�����������������������������������������������������������������������TTZabffhca`TTOOPTTTT������������������������������������������������������������fgjrt{���������tlgef35;BNRNGB:54333333339BFOV[cggd[OLIDB<:99;>==?CDHTW\]\ZTQH@;;tz������zwsrtttttttt��
���������)5[�����������k`ZB))����
##$#
����8Hanz�������xnUH<528�����			����������05?BCCB5440000000000dht���������������hd����������������������������������������MTWabcaTPMMMMMMMMMMM�����
����������
#-.#
����������������������������������������������������������������������������������������*6CLTWWTOC6**6BO[ehiga[YOB63)"�����������x�����������~xxxxxxx���������������������������������������������������������������	(313-)���BHO[[]htutsnjh[QOB7B����������������������������������������#0<ILOMJI<2+# ##0<IIKLJIA<60'#""##������� �����������
###
����"$#'������06BJOQWUOB@634000000>BGN[^gtwtigf_[NB>>>Y[gt��������tgc[TQYY�����������������������������������������������������������������������������������������������������"	 ��������)5B]`NEB5)������������������������������������).58BN[bgokg[NHB5.))����#)5:?:�����UUabnn{������{nbZUUU!#/<FE?<8/#!!!!!!bbn{��������{ronkbbb��������������������Zgt������������tg[XZ��������������������#&&%%$## ���������������������������������������޼������������������üȼʼʼʼ������������g�d�[�Q�C�;�D�N�[�t�t�gE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E���������������	������������������������ݽսؽݽ���������������������U�O�U�U�a�a�n�x�z�t�n�a�U�U�U�U�U�U�U�U�s�f�M�A�$�&�,�6�A�M�a�s���������������s������ĿĻĽĿ�������������������������ؼ���������� ������������T�;�1� ��!�;�T�i�y���������������y�m�T�Ŀ��Ŀȿѿݿ����ݿѿĿĿĿĿĿĿĿ�àßÓÓÒÓÖàìòù����������ù÷ìà�H�>�;�>�H�Q�T�W�U�T�H�H�H�H�H�H�H�H�H�H�6�4�1�6�7�B�M�O�[�\�[�O�E�B�6�6�6�6�6�6�6�2�4�6�B�N�O�[�O�B�6�6�6�6�6�6�6�6�6�6�)�&�'�)�)�*�6�6�7�6�)�)�)�)�)�)�)�)�)�)�/�,�"�!��"�%�/�;�A�H�T�X�T�Q�M�H�;�/�/����������������������������������������N�8�(���(�0�A�N�Z�g�m�s���������s�g�N���������������������������������������y�m�^�\�`�j�m�y�����������ſǿſ�������������������������������������������������������������������������������������ƳƲƳƸ������������������������ƳƳƳƳ�g�f�Z�N�L�C�N�X�Z�Z�g�j�s�g�g�g�g�g�g�g�����������źɺֺ׺����ںֺɺ���������������ľĽĶĿ������������� �����������C�=�6�*�%����*�-�6�C�P�\�e�c�\�U�O�C������������������������������������������|����������������ļü����������������������s�g�Z�E�D�Z�s���������������������������)�2�-�)�������������������Ǿʾ׾������������׾ʾ���²®��l�q¦������-�W�_�H�/�
����²�$���������$�(�0�=�I�L�I�=�:�0�$�����{�y�{�����ùܹ� �����߹չƹ������"���	����	���"�,�&�"�"�"�"�"�"�"�"�a�U�a�i�m�z��{�z�m�a�a�a�a�a�a�a�a�a�a�û����������s�o�r�x���������ûֻݻڻѻý����������������Ľнѽܽ۽ؽн̽Ľ������U�a�h�n�o�n�m�h�a�_�U�H�>�<�8�:�<�H�T�U����!�"�/�8�8�/�"����������Ň�~�{�v�t�w�{ŅŇŔřŠţŦŦŤŠŔŇŇ�������������ſѿݿ��	��(�4�(�����ѿ��x�_�N�O�S�\�l�������������ɻϻӻ»����xìéàÓÑÇ�{ÇÓàçìùù������ùìì�/�.�"���$�/�<�H�U�Z�a�n�x�n�a�U�H�<�/¦¦¬²¸¿��¿²¦¦¦¦¦¦��׾ҾϾо׾����	��"�,�+�&�"��	�������������������������������������������������x�l�l�r��������������޼ּ��������������� ������������������A�8�>�D�N�P�Z�g�s������������s�g�Z�N�A�����������ʼʼʼѼʼ�������������������ƚƖƎƁ�vƁƎƚƧƩƧƜƚƚƚƚƚƚƚƚ�f�a�Z�Q�M�C�A�A�M�Z�f�s����������u�s�f�͹ù����������¹ùǹϹܹ������ܹͺL�A�?�D�K�Y�e�r�����ֺ�޺׺ĺ����~�Y�L������������!�!�!���������ｒ�������������Ľݽ��ݽݽ�нĽ�������������������� �(�*�(�#��������D�D�D�D�D�D�D�EEEE*E0E*E#EEEEED�E7E2E7E?ECEPEPEPEYEPEMECE7E7E7E7E7E7E7E7�`�V�:�)��.�;�S�l�y�����������������l�`���������������ĽʽŽĽ������������������)�"����������#�)�5�9�?�;�5�)�)����������	��������	������׾ʾ��������ʾ����	����	��������������������������������������������������������������������*�6�<�7�*�����*��������*�6�=�C�H�C�@�6�*�*�*�*��v�~�{�����������ʼ����м����������������������������������ôìÛÔÕÙìù�����������-������Ň�}ŇŎŔŠŢŧŠŔŇŇŇŇŇŇŇŇŇŇ�S�K�S�]�s�|�����������������������s�g�S�����������������������������������������[�W�J�D�D�[�h�m�~āčĚěĖĔėćā�h�[���������s�s�m�p�s����������������������E�E}EuEzE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��������'�4�@�M�M�M�H�@�4�0�'����#���
��
���#�0�1�<�=�=�<�0�#�#�#�#ĚčĉĆĈčĚĳĿ����������������ĳĦĚ������������$�0�-�$����������������x�w�t�x�������������������������x�x�x�x A 1 2 3 U 6 K H .   ! _ 7 d O ; z * J 6 W = J ; i Q 0 J 9 l a � \ \ F < 6 q n z = 8 Z 7 D , � > b  = % h M z o # 6 Y @ 3 $ Y r ' I g 8 & U S , T i P : c p T Q @ M 8 - U u  �  �    x  �  �  �    �  `  v  �  E  �  �  ^     �  "  �  k  �  m  1  �  �  �  S  .  V  !  2  �      *  -  r  (  K  �  #  O  ^    "  �  �  �  �  �  Z  �  z  M  X  �    �  �      )  f  �  �    �  �  �  z  �  �  �  �  8  ^  Q  �    �  �  �  |  �  �;�o:�o������`B��`B�t�����\)��t��e`B��hs��o�C���1�ě���`B��9X��9X�',1���ͽ8Q켴9X��j��/���ͽ\)�m�h�\)��`B�+���C���㽬1�'��P���o�e`B�0 ŽY��\)�P�`��o��7L�49X�y�#�\)�y�#�P�`��t��#�
�H�9�#�
�,1�ixսm�h��\)�D����%�<j���P�H�9���-�49X�@��q����Q�8Q콅��L�ͽq���@���vɽaG���1�y�#���w�m�h��{�}󶽉7L��^5�������B�>B$�YB�sB�]B��B�B��B�yA�B��B�.B �$B��B�B�_B�7B� B��B�|BtB$�B*�%B�B%�A��Bw�B!y�B zB
UB��B�[A�@lB hBwB	�iB�bB{�A��/Bi�B��B!)�B!?�A��lB��BZ;B�BF�B�pB̽B0-TBs,B+B
�6B)B��BE�Bw`B/�B	:B��B%��B&@BtZBB_�B;�BM"B	��Bl�BօB�B��B,lB.��By�BT�B=3B�B�aB(^�B;B(�6B�B
nVB.LB%B��B$��B �B��BBfB2�B�B>�A�}B�!B�B �&B��B �B��B«BD�B��B��B�\BCB*�B9wB?oA�{�BA�B!DBHuB	��B�1B��A��GA�c5B>B
@�B?�B�A�~BA=B 0~B!��B!>GA���B�B>�BAbB?hBǣB�!B0@�B��B=?B
}�B:2B�BJB=�B=>B��B�B%��B&=RBA�B@}B^�Bo�B>�B	��B�B��BCAB�BB,B�B.��B�FBB�B?]B=�B �B(�SB?`B(��B�B	��BC�B$�|A�%@��;A��mC�T�A�#A-�qA��A@`EA�r�@�S Ai��A|�+A�f}A�_^Aؔ8A؎A֠A�Q�A���A��HA�E�Ap@�A���A�QFBnoA��!@6l�A��$B �#A�JN@�:�A�ȤA��;AT#A���B	�=z/�A�xUA�]%@�<A%�WAň�A��A��A}܃@�*�A��"A�P&A���AX��A��u@�:�A�ߌA��?@�,4B�7AA7�>��R@E@W��A#��A1�vC�Y�C���A<�A#��A���AYF�AUH5A�A��A���@��AA�=�A��A��tA��A�>�A�8�C��@�`A��hA�xB	M@�6�A�ou@��mA�z�C�U?A���A.��A�~�A=*A��b@�0�Ak0A~]XA̔AA�t�A�r�A؁A֍aA�AA�p�A�upA�|�An�oA��aA���BEA�n�@;mbA䞄B7<A��@���A�(�A�9�AS;�A�~AB	��C���A���A��o@���A&n^A�|�A��A�?�A}O:@��A�7;Ač�A�jiAW_cA��@�K�A�c�A��V@��B�<AA"�>���@��@[}�A"��A4(�C�V�C��PA��A#�oA��AZ cAU;A��)A�yB *B@�PA	�A�P�A�t�A���A�v�AۦA��hC�%@�DnA�(A�r�B	�@��%      	                      
      =                                                   $         
      
      ;      0                        "   %                  '                        	         "   	   %            1                  1      !   
                        
                        %         '                        %   !      #                                       A      #         %               !   %                  %                     '                  #            !            !      +      %                                                   !                                 %         !                                       -      #         %                  #                  !                     '                  #                        !      +      %                           N_��N���O*ŻNp0�N�jN�g�N[E_O��9N���NNBO���N_�O �NM�NdU�NDM�-�N��Oޯ�O}ϚN3|�O�ؖN;X�N�5N�Z�Nl�;N�"�OsvwO"�NZzN���O4� N��N��hP BvO�O�??N>�2M���O�VlOfOc1N��O,O�aAO̙9N��gOd]N��OS�MO,�wO؟�NX³N�v=MᤋN<�O;>1N(�Pj N���O�tN���Nʘ�NPDOѫ�N�/NZh�O)/O��N�DFO8�tN�9�O��NK-�P�fN);�O��N��O���N���N�@�N��!NW�O���N��Nl��  	  �  \  �  &  /  k  T  �  �  )  ,  e  2  �  �  �  E  Z  �  `  `  �  �    E  c  �  2     �  	  �  9  q    �  �  �  �  ^  �  �    �  <  +  n  5  _  �  *  �  @  �  �  �  c  �  �  �  �  	�    B  K  *  �  )  �  �  �  
  �  �  1  �  �  �  !  	  O  �  �    �<o<t���o;D����o�D����`B�#�
�t��t����ͼe`B�e`B��o��C���C���C���C�����ě����㼣�
���
���
��1��1��1������j�ě��ě����ͼ��ͼ�/�49X������`B��`B��`B��`B��h��h���+��P�\)�o�o�o�t��C��#�
�\)���t�����w�D���#�
�#�
�<j�#�
�,1�',1�'0 Ž49X�T���,1�D���49X�0 Ž0 Ž8Q�P�`�T���T���Y��Y��y�#�aG���%��+��O߽\sz������zwqssssssss�� 
#'.#
	���������
$'*'#
 �����������������������������������������
###
������ABFOY[c[OB=8AAAAAAAA���
#<HMNH7/#
��-/18;HKRRSPHB;0/----��������������������6B[hmov}xj[OB;6/+,/6����������������������������������������0<HU^UQH<90000000000-/<GHUHG<<;6/)------}�������{}}}}}}}}}}��������������������%)36BNOPPOLHB:6)&!%%/H\amy�����znlU</&%/<?CHUanz��zunaUHC<<�����������������������������������������������������������������������TTZabffhca`TTOOPTTTT������������������������������������������������������������fgjrt{���������tlgef35;BNRNGB:54333333339BFOV[cggd[OLIDB<:99;>==?CDHTW\]\ZTQH@;;tz������zwsrtttttttt������������N[t�����������tfcZPN����
##$#
����8Hanz�������ynUH<538�����			����������05?BCCB5440000000000dht���������������hd����������������������������������������MTWabcaTPMMMMMMMMMMM���������������
"&#
����������������������������������������������������������������������������������������%6CHQSSOHC6* -6BO[chhf`[WODB6)) ����	�������x�����������~xxxxxxx���������������������������������������������������������������	(313-)���JOP[hjjh\[YOJJJJJJJJ����������������������������������������#0;<FHHGC<0####0<IIKLJIA<60'#""##��������������������
###
����!#"$������06BJOQWUOB@634000000?BJN[^`][NB?????????S[\gt��������tgf[VSS�����������������������������������������������������������������������������������������������������"	 ��������)5B]`NEB5)������������������������������������).58BN[bgokg[NHB5.))����#)5:?:�����UUabnn{������{nbZUUU #&/;<@<<5/#       bbn{��������{ronkbbb��������������������Zgt������������tg[XZ��������������������#&&%%$## ���������������������������������������Ҽ������������������üȼʼʼʼ������������s�g�Z�N�K�E�N�[�g�t�u�sE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E���������������	������������������������ݽսؽݽ���������������������U�S�U�W�a�d�n�u�x�r�n�a�U�U�U�U�U�U�U�U����s�Z�M�A�,�*�3�:�A�M�\�s������������������ĿĻĽĿ�������������������������ؼ���������� ������������?�6�.�0�;�G�`�y���������������y�m�`�T�?�ѿǿοѿݿ����ݿѿѿѿѿѿѿѿѿѿ�ìààÔÓÔØàìðù����������ùõìì�H�>�;�>�H�Q�T�W�U�T�H�H�H�H�H�H�H�H�H�H�6�4�1�6�7�B�M�O�[�\�[�O�E�B�6�6�6�6�6�6�6�2�4�6�B�N�O�[�O�B�6�6�6�6�6�6�6�6�6�6�)�&�'�)�)�*�6�6�7�6�)�)�)�)�)�)�)�)�)�)�/�,�"�!��"�%�/�;�A�H�T�X�T�Q�M�H�;�/�/����������������������������������������g�Z�N�D�1�.�2�<�A�N�Z�_�g�s�~�������t�g�������������������������������������������y�m�_�]�`�m�y�����������¿Ŀ¿�����������������������������������������������������������������������������������ƳƲƳƸ������������������������ƳƳƳƳ�g�f�Z�N�L�C�N�X�Z�Z�g�j�s�g�g�g�g�g�g�g�����������źɺֺ׺����ںֺɺ���������������ĿĿĿ���������������������������C�=�6�*�%����*�-�6�C�P�\�e�c�\�U�O�C������������������������������������������|����������������ļü����������������������s�g�Z�E�D�Z�s���������������������������)�2�-�)�����������ʾž��ʾҾ׾ھ��������׾ʾʾʾʾʾ�¿¢¢°�������
��#�H�P�N�F�/�
����¿�$���������$�(�0�=�I�L�I�=�:�0�$�����|�z�}�������ùܹ�����޹ԹŹ������"���	����	���"�,�&�"�"�"�"�"�"�"�"�a�U�a�i�m�z��{�z�m�a�a�a�a�a�a�a�a�a�a�û����������s�o�r�x���������ûֻݻڻѻý����������������Ľνн۽ڽֽнʽĽ������U�a�h�n�o�n�m�h�a�_�U�H�>�<�8�:�<�H�T�U����!�"�/�8�8�/�"����������ŇŁ�{�w�u�y�{ŇŇŔŖŠŢťŤšŠŔŇŇ�Ŀ����������ѿݿ������������ݿѿĻx�_�Q�R�V�_�l�x���������Ļʻʻû������xìéàÓÑÇ�{ÇÓàçìùù������ùìì�/�.�"���$�/�<�H�U�Z�a�n�x�n�a�U�H�<�/¦¦¬²¸¿��¿²¦¦¦¦¦¦��޾׾ѾӾ׾�������$�&���	�������������������������������������������������u�o�o�r�|�������ּ����ּ������������������� ������������������N�J�L�N�Z�]�g�s�z�������������s�g�Z�N�N�����������ʼʼʼѼʼ�������������������ƚƖƎƁ�vƁƎƚƧƩƧƜƚƚƚƚƚƚƚƚ�f�a�Z�Q�M�C�A�A�M�Z�f�s����������u�s�f�ϹĹù����ùϹԹܹܹܹٹϹϹϹϹϹϹϹϺL�A�?�D�K�Y�e�r�����ֺ�޺׺ĺ����~�Y�L������������!�!�!���������～�������������������ĽнֽнŽĽ�������������������� �(�*�(�#��������D�D�D�D�D�D�D�D�EEEEE"EEEEED�D�E7E2E7E?ECEPEPEPEYEPEMECE7E7E7E7E7E7E7E7�`�W�:�+�!�.�=�S�l�y�����������������l�`���������������ĽʽŽĽ������������������)�$�����)�5�7�=�9�5�)�)�)�)�)�)�)�)������������	��������	�����׾ʾ������ľʾ׾����	����������������������������������������������������������������������*�,�+����������*�&������"�*�6�:�C�E�C�>�6�*�*�*�*��v�~�{�����������ʼ����м����������������������������������ôìÛÔÕÙìù�����������-������Ň�}ŇŎŔŠŢŧŠŔŇŇŇŇŇŇŇŇŇŇ�S�K�S�]�s�|�����������������������s�g�S�����������������������������������������[�W�J�D�D�[�h�m�~āčĚěĖĔėćā�h�[���������s�s�m�p�s����������������������E�E�E�E{E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��������'�4�@�M�M�M�H�@�4�0�'����#������#�-�0�:�8�0�#�#�#�#�#�#�#�#ĚčĉĆĈčĚĳĿ����������������ĳĦĚ������������$�0�-�$����������������x�w�t�x�������������������������x�x�x�x E 1 ' 3 U 6 M H .    X 7 d O ; z * J 7 W 8 J ; i Q 0 @ 9 l a � \ T ^ < 0 q n z : 8 Z 7 ) ) � > b  9 & h B z o # A Y @ + $ > r ' I / 6  U R > T i P : c p T Q < M 1 - U u    �  o  x  �  �  |  �  �  `  �  ,  ;  �  �  ^     �  "  �  k  �  m  1  �  �  �  �  .  V  !  2  �  �  '  *    r  (  K  r  #  O  5  A  �  �  �  �  �  �  �  �  �  M  X  �  R  �  �  R    �  f  �  �  k  U  �  �  �  �  �  �  �  8  ^  Q  �    �  �  ]  |  �  �  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  FH  �        	        �  �  �  �  �  �  �  y  i  \  N  A  �  �  �  �  �  �  �  �  �  �  �  x  i  X  G  7  &    �  �    $  :  K  W  Z  \  T  F  0    �  �  \  �  �  �  b  �  '  �  �  �  r  _  L  :  ,      �  �  �  �  �  �  �  �  ]  -  &    	  �  �  �  �      "  5  J  _  k  e  _  X  Q  J  C  /  )  "              �  �  �  �  �  �  z  O  $   �   �  /  H  ]  m  t  x  {  }  ~    ~  {  s  l  c  Z  R  L    �  ?  M  S  P  E  6  $      �  �  �  �  �  �  S    �  1  �  �  �  s  _  K  3      �  �  �  �  �  �  �  �  �  �  �  w  �  �  �  �  �  �  �  �  �  �  t  f  S  <  &    �  �  �  �  �  �      )  (  $     $    �  �  �  �  Z    �  $  �  �       "  $  '  )  +  ,  ,  +  +  +  +  +  ,  -  .  0  1  2  d  e  _  U  I  :  '    �  �  �  �  f  1  �  �  e    �  ,  2  .  )  %        �  �  �  �  �  �  �  �  o  [  E  0    �  �  �  �  �  y  a  G  -    �  �  �  T    �  6   �   �   �  �  �    �  �  �  �  �  �  �  a  >    �  �  �  z  O  "   �  �    (  K  o  �  �  �    @  t  �  �    Q  �  �  �    I  E  ;  0  &         �  �  �  �  �  �  w  k  ^  K  7  #    Z  Y  V  S  L  D  5      �  �  �  �  �  �  v  I      �  i  �  �  �  �  �  �  p  \  H  ;  6  +    �  �  �  (  �    `  T  I  =  -        �  �  �  �  �  �  q  P  .  
  �  �  W  `  ^  \  ]  ]  Y  T  P  H  <  &    �  �  n  1  �  C   �  �  �  �  �  }  v  p  i  c  \  U  L  D  <  3  +  "      	  �  �  u  j  ^  S  G  7  #    �  �  �  �  �  �  �  u  b  O      �  �  �  �  �  �  �  �  �  �  ~  q  d  W  I  4    
  E  ;  2  (        �  �  �  �  �  �  �  �  f  F  &     �  c  [  R  H  >  +    �  �  �  �  `  >    �  �  �  �  �  �  R  e  �  h  H  '    �  �  �  �  Z  !  �  �    �    �    2  ,  &                �  �  �  �  �  �  �  �  x  `     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  [  F  1  �  �  t  h  [  O  B  3  "    �  �  �  �  �  |  r  f  M  4  	     �  �  �  �  �  �  p  \  G  1    
  �  �  �  �  �  �  �  �  �  �  �  q  U  8      �  �  �  �  �  �  �  �  �  �      )  9  8  5  1  (      �  �  �  j  -  �  �  c    �  �  �    ,  E  g  p  i  j  `  I  '  �  �  /  �    �  �  �    �  �  �  �  �  �  }  e  N  5  '    �  �  �  �  x  K    s  �  �  �  L    �  �  �  l  I    �  �  ^  �  �  �  O  e  �  �  �  �  �  w  i  U  >  '    �  �  �  �  �  �  o  X  A  �  �    "  5  B  A  ?  >  <  8  0  (           �  �  �  �  �  d  B  +  �  �  l  E    �  �    G  �  �  �  �  -     Z  ]  ]  [  V  J  7  "      �  �  �  �  �  �  w  `  S  _  �  �  �  �  j  A    �  �  �  ]  )  �  �  �  1  �  +  t  �  �  �  �  �  �  �  �  �  �  �  �  w  g  T  A  .      �  �  �  	      �  �  �  �  �  e  ,  �  �  o  5  �  �  s  %  ^  p  �  �  �  �  �  �  �  k  J  %  �  �  �  k  &  �  _  �  �    5  <  1    �  �  �  a  J  =  %    �  �  F  �  !  P    +  	  �  �  �  �  k  E    �  �  {  �  �  �    �  8  �    n  T  9    �  �  �  �  W  )  �  �  �  6  �  r  �  ,  a    5  -  $           �   �   �   �   �   �   �   �   �   �   �   �   �  J  W  \  ]  [  T  H  6      �  �  �  k  6  �  �  #  �  !  �  �  �  �  �  �  �  �  �  o  Z  E  -    �  �  �  t  V  9  �    $  *  "    �  �  �  A  �  �  J  �  �  I  �  �     �  �  �  �  �  �  �  �  �  u  g  X  I  ;  .         �  �  �  +  &  1  ?  @  >  9  2  *      �  �  �  �  ^    �  i    �  �  �  �  �  �  �  �  �  �  w  l  b  W  L  B  7  -  #    �  �  �  t  `  K  4      �  �  �  �  �  m  M  ,     �   �  �  �  �  �  �  g  H  (    �  �  u  C    �  �  �  B  �  m      #  )  8  E  P  Z  a  c  ^  R  C  -    �  �  �  G  �  �  �  �  �  |  S  ?    �  �  ~  G    �  �  �  j    �   �  �  �  �  �  �  �  v  _  A  !     �  �  �  j  F  #     �   �  i  �  �  �  �  �  �  �  �  �  �  u  Z  4  �  �  O  �  d   �  �  �  �  �    o  _  I  2    �  �  �  �  �  _  @  +      	!  	�  	�  	�  	�  	\  	,  �  �  ~  2  �  �  O  �  �  X     x  �    �  �  �  �  �  s  ]  M  ?    �  �  b  &  �  �  i  &  �  A  <  1  "    �  �  �  �  �  �  �  �  �  �  C  �  `  �  2  K  F  @  :  5  /  *  #          �  �  �  �  �  �  �  x  '  (  )  )  *  *  *  +  %      �  �  �  �  q  S  7     �  �  �  �  �  �  �  �  �  �  b  @    �  �  }  A    �      �  �    $  )  $      �  �  �  �  L  �  �    �  �  �  �  �  �  �  �  �  �  �  �  {  r  j  b  Z  L  3       �   �   �  �  �  �  �  �  �  �  �  �  �  �  s  S  2    �  �  �  w  5  `  m  z  �  �  �  {  s  k  _  R  E  3  !      �  �  �  �  
    �  �  �  �  �  �  ~  _  >    �  �  �  �  �  b  3   �  �  �  y  l  `  S  C  4  %      �  �  �  �  �  �  }  b  G  �  �  �    a  ;    �  �  x  [  K  %  �  �  U  �  i  �  �  1  -  *  &  #          �  �  �  �  �  �  �  �  �  �  ~  �  |  \  7    �  �  s  6  �  �  l    �  W    �  Q  �  �  �  �  w  k  ^  N  <  ,      �  �  �  �  �  z  b  I  .    �  �  l  O  5  .    �  �  �  a  @  1    �  �  �  E    J  !      
    �  �  �  �  �  �  �  �  r  Z  B  +     �   �  �  �  �  	  	  	  �  �  �  �  h  +  �  O  �  !  u  �    S  O  C  6  ,  %          �  �  �  �  �  �  |  `  B  $    �  �  �  �  �  �  �  �  �  �  �  �  k  Q  7        �   �   �  �  �  �  l  _  b  \  F  "  �  �  �  �  �  �  _    �  -  �        
     �  �  �  �  �  �  �  k  L  +    �  �  �  �  �  �  �  �    v  o  m  n  z  �  �  �  �  w  m  h  c  b  `