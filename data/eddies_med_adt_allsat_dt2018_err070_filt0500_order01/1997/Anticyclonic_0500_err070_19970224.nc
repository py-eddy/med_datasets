CDF       
      obs    8   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?ɺ^5?|�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�y�   max       P��j      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��h   max       =��      �  l   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>Ǯz�H   max       @E�          �   L   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�=p��
    max       @vl�\)     �  )   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @P@           p  1�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�q        max       @��           �  2<   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��1   max       >{�m      �  3   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B,      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B,BH      �  4�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?sK   max       C�)�      �  5�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?S/�   max       C�8$      �  6�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;      �  8\   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          '      �  9<   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�y�   max       O�MA      �  :   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�,<�쿲   max       ?�ح��U�      �  :�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��`B   max       >�+      �  ;�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�=p��
   max       @E�          �  <�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��
=q    max       @vl�\)     �  E|   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @P@           p  N<   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�q        max       @�2�          �  N�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         CE   max         CE      �  O�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���p:�   max       ?��
=p��     �  Pl   
         	            4      N            %         �      B   	      {      -                           V   $      
            �   )         	      *         c            /   )      N�62Ni�	O�DNYl�O�̊NZGO;�O���N���P�&Oi%N�Nh�dO���N�q|OiY�P?[BN-&�O��N�"�O�O�M�N�ͦPno@N�mXPB�O���N�N�z�Oq�;N�xOc~P)��OT��N(6�NOs7N��N�<�OA�P��jO}$UO[�GN���N�h�NfRO���N��SN�O��?N��M�y�O?8O�_�O���O5}&NVR��h��`B�#�
�t��o;o;ě�;ě�;�`B<#�
<D��<D��<�C�<�C�<�C�<���<�1<�j<�j<�j<���<�/<�/=o=o=C�=C�=t�=t�=�w=�w=,1=0 �=0 �=0 �=49X=49X=49X=8Q�=<j=@�=@�=@�=D��=D��=H�9=H�9=P�`=Y�=Y�=Y�=e`B=ix�=�C�=�t�=��qjhmqt��������xtqqqqBBGLNO[^bb[YOBBBBBBB���������� ����������������������������QPOPT^mz~�����zmaXVQ35BNUXNB:53333333333�}~����������������
/<HPRU]^UH/#��������������������JIQWht��������th[VQJQOTY[_gt������utgc[Q��������������������st��������vtssssssss����������������������������������������eadmx�����������znme" ")6BOq|��thgVB>?."��������������������W[g������������|vrgW��%(&"����	"/;BEE?6"
��������

�����713<HSUaUNH<77777777�)[g~���{tjaN+)/5BN[]^[ONB;5))))))#05JMV[\YXOB)05BFK[gt����tg[NB50�������������������������������������������
#/354.'#
���#)/15/,#)6BFDB>/)&�����������������������#"����#%030/010##��������������������ddhmtxzwthddddddddddrnotw}���������trrrr""/;EGHTWYZXTOH;/"z{����� "������z�������"&(#���������!" �� ����      ����
#&'#
�������������������������v~����������������zv 
#(,)#
������������������������� +.-&����").56752)!#/<<<5/#snllot~����������xts����������������������*275-����������

�����
����
�H�U�a�n�z�Ä��z�n�a�`�U�N�H�D�H�H�H�H�y�z�����������������|�y�q�w�y�y�y�y�y�y������(�3�9�6�4�(�����޽ӽнӽݽ��
�����
���������
�
�
�
�
�
�
�
�
�
�����������	������������������������������������������������������������������������������������������������������̿m�y���������y�m�`�R�;�.�-�)�&�*�;�J�T�m�N�Z�g�r�s�g�c�Z�R�N�N�L�N�N�N�N�N�N�N�N��������ϾӾӾʾ���~�s�f�G�<�9�A�M�Z���*�6�C�I�\�h�l�c�\�C�6�*��������f�s�����x�s�f�b�]�f�f�f�f�f�f�f�f�f�f�Y�Z�c�e�g�e�Y�L�@�A�L�R�Y�Y�Y�Y�Y�Y�Y�Y�������'�3�=�>�:�3�'������ܹٹڹ�ùù��������ùìçà×ÓÐÓÜàìöùùĚĦĳļĿ��ĿĻĮĦĚĖčĊćąāčĎĚ�Y�f������������r�@�4�����ܻѻܼ�4�Y�����������������������������������������O�^�b�Z�Y�T�L�6�)��������������2�O���������������������y�r�y�y�������������"�/�;�H�T�a�c�h�k�h�a�T�;�/�"��	��	�"D{D�D�D�D�D�D�D�D�D�D�D�D�D�DzDlDhDfDoD{������������������������������������/�H�\�Z�J�=�5�/��������������������	�/�Ľн۽ٽ׽ҽнĽ������������ĽĽĽĽĽ���������=�[�=�0�$��������ƸƛƒƠ���������������������������}�s�t�v�q�n�t��/�<�H�U�a�]�U�H�E�<�6�/�-�%�%�.�/�/�/�/������������������ܻ޻�����N�[�g�t�}��t�q�g�[�N�B�5�)�$�"�'�5�;�N��$�!�������������������e�r�������������������~�r�Y�B�L�T�Y�^�e������� �������ùìàÇ�p�yÄÔñù���H�U�a�k�m�j�l�d�a�U�H�<�/�(�#���#�/�H�������������������������������������������������������������������������������һl�x���������x�l�d�a�l�l�l�l�l�l�l�l�l�l���������ûлֻػлû����������������������
���)�:�I�U�X�R�I�<�0�#���������������ż������ּʼ�����l�\�I�M�f����!�-�:�F�S�]�S�F�C�:�-����������׾����	��"�'�)�"��	�����׾Ӿɾʾ׻l�x�����������������y�x�l�a�l�l�l�l�l�l���������Ⱦʾ��������������������������������������������y�v�y������������������5�A�N�b�m�s�z�s�g�Z�N�5�(�"�����%�5�ּ��������������ּ˼ʼʼʼʼּּּּ������ż�������������������������������������#�@�R�U�3�#����������ļĽ���������b�n�{ŇŌŔŝŠŤŠŗŔŇł�{�n�b�a�V�b¢����(�5�:�A�C�H�?�5�(������������������������������������{�z�{�z�}���¦²¿��������������¿²¢¦E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�ExE�E��I�H�C�I�O�U�b�g�j�b�a�U�I�I�I�I�I�I�I�I ; U : 2 A t N ? * * = 1 3 7 = 7 T X K G A  B _ S \ N O < # � @ - + R > P ` f O . 8 T " p \ " ) P N c 2 8 1 X .    �  ~  c    i  ?  �  �  �  �    �    �  �  �  T    �  N  �  �  �  �    *  �  �  �  o     �  �  c  n  ;  �  �  �  �  �  �  �  i  �  �  '  g  7  6  Y  �  i  �  l���㼬1<�o%   <o;�`B<ě�=m�h<49X=�-=�w<u<�h=Y�=8Q�=#�
>�<�h=�9X<��=L��>�u=,1=���=C�=�o=D��=aG�=aG�=�7L=,1=�O�>�=��w=@�=]/=T��=�C�=e`B>{�m=�-=�hs=u=ix�=Y�=�Q�=ix�=ix�>��=��P=e`B=�C�=��=�;d=��=�S�B
�B�B"@ZB(�A��B��B ��BXiB��BB_B	��B!p�B�B��B""B X+B��B�B
��BZ�A���B�SB'
BPjB�
B�B� BߜB �TB=bB�QB2�Bu�BI�B%9�BΌB�B��A�`�B
LB��BڊB�B$vRB,B��B$�DB#P�BsBm
B!B
l:ByXB}�B%�B��B
 B��B"�ZB<�A��BxB ��B��B�B>�B	�B!SBǁB�LB"?�B �B��B�pB
�tB��A��B��B.wBǎB��B�B	A5B�B!0�BBLB��B>�B��BF*B%?�B��B;�B��A�inB62B��B�B�B$?�B,BHB9B$خB#?BF[B�@B?�B
^iB��B>�B��B��Aƻ�AПA0�A��A���A��B1�AgܟA��nAF��B {}ABq�?���?sKA��HA�@-@�A� tAֶ�Aq�A��AC�ֵAҧA��A'�(B�[AF�zAë�@�R�A�QA�z�@�AϕZAĔ=@�\A�@��m@�I-A��@���@l�uAXr�@�]�AL��A��A���A�@�P�A穑A�,KA�A�B�A��A��C�)�A�)A�y�A �A1)�A��MA�[A�|�B?�Ag �A��AF�B BAA�?��?S/�Ā�A�L@ݴ�A�Y8A֋�Aod,A�C�C��SA�~JA��#A'@�BVAH�A�@@��A���A��	@��A�}�A�|�@��A�g�@��"@�N�A��@���@k�AX C@�AL��A}A��A>p@� �A�w�A��A�~�A�lA��A��xC�8$A�~3            
            5      N            %         �      B   	      {      -                           W   $         	         �   )         
      *   	      c            /   *       	                              )                     3      %               ;      1                     +                     ;                           %            !                                                            !                     '      '                                                                                 !         N�U�Ni�	OPd�NYl�O=��NZGN�(O���N7�,O�\�O0W�N�Nh�dO&m@N�q|OiY�O��#N-&�OclGN�"�O�ǼOX�N�ͦO�MAN�mXO�cO���N�
N�z�O9��N�xOcg�O�{|O8MN(6�NOs7N��N$�4OA�O�w�OP�O[�GN���N�h�NfRO_2�N��SN�O��rN�aM�y�O
`�O�_�O�+�O5}&NVR  �  �  �  �  �  �  �  �  �  �  !    r  T    v  X  �  �  z  �  �  ?  �  �  i    �  I  V  �  &  	�  �  �  �  �    �  }  I  t  �    �  �  W  �  ~  �  �  7  g  �    b��`B��`B�D���t���o;o;�`B<�o<o=�w<�o<D��<�C�<���<�C�<���=�C�<�j=49X<�j<���=� �<�/=@�=o=��=C�=��=t�=49X=�w=49X=��P=<j=0 �=49X=49X=Y�=8Q�>�+=T��=@�=@�=D��=D��=aG�=H�9=P�`=��=aG�=Y�=ix�=ix�=���=�t�=��rkinqt��������vtrrrrBBGLNO[^bb[YOBBBBBBB����������������������������������������RT[amrzz~��~zrma^ZUR35BNUXNB:53333333333�~~����������������#/<AKQTVUH</# ��������������������UTVZ`ht��������tha[U[Y[[]`dgt�������~tg[��������������������st��������vtssssssss����������������������������������������eadmx�����������znme0./16BO[hnsvurh[OB60��������������������sjrt��������������ys��%(&"����
"/;@DD>5/"
�������

������713<HSUaUNH<77777777)5N[t���tg[NB5)/5BN[]^[ONB;5))))))	&5BDGRXVTIB)	05BFK[gt����tg[NB50������������������������������������������
#*.0/*#
��#)/15/,#)6BDCA=64-)��������������������������! ���#%030/010##��������������������ddhmtxzwthddddddddddtt{���������zttttttt""/;EGHTWYZXTOH;/"�������������������������#$$���������!" �� ����      ����
#&'#
��������������������������}������������������ 
#(,)#
�������������������������#&((&����%),55651)'!#/<<<5/#ommqt�����������too����������������������$-30)����������

�����
����
�H�U�a�n�z�~Ã�}�z�n�b�a�U�P�H�F�H�H�H�H�y�z�����������������|�y�q�w�y�y�y�y�y�y�������#�(�)�(��������۽ڽݽ���
�����
���������
�
�
�
�
�
�
�
�
�
�����������������������������������������������������������������������������������������	��������������������������̿`�m�y��������y�m�T�G�;�9�/�-�-�5�;�T�`�N�Z�g�n�o�g�[�Z�Z�P�N�N�N�N�N�N�N�N�N�N�������������������������s�f�S�R�Z�g���*�6�7�C�O�\�a�g�^�\�O�C�6�*�!�����f�s�����x�s�f�b�]�f�f�f�f�f�f�f�f�f�f�Y�Z�c�e�g�e�Y�L�@�A�L�R�Y�Y�Y�Y�Y�Y�Y�Y�����!�'�0�3�6�3�2�'������������ùù��������ùìçà×ÓÐÓÜàìöùùĚĦĳļĿ��ĿĻĮĦĚĖčĊćąāčĎĚ�Y�f�r����������z�r�f�M�@�4�"���,�E�Y�����������������������������������������)�B�O�O�P�P�J�B�@�6�)����������)���������������������y�r�y�y�������������"�/�;�H�T�a�b�h�k�f�a�T�;�/�"���	��"D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D}D�D�D�D��������������������������������������"�/�;�;�7�7�2�&��	���������������	��Ľн۽ٽ׽ҽнĽ������������ĽĽĽĽĽ������������<�@�=�0�$���������Ƥƛƭ�̾�����������������������}�s�t�v�q�n�t��<�H�U�Y�U�U�H�C�<�3�/�.�'�&�/�1�<�<�<�<������������������ܻ޻�����N�[�g�t�x�t�m�g�[�N�B�5�*�)�&�)�-�5�B�N��$�!�������������������e�r�������������������~�r�f�e�Y�N�Y�`�e������������� ������������ùïçåæø���/�<�H�U�a�j�i�j�a�a�U�H�@�<�/�*�#�#�/�/�������������������������������������������������������������������������������һl�x���������x�l�d�a�l�l�l�l�l�l�l�l�l�l�����ûͻлһлû��������������������������
���)�:�I�U�X�R�I�<�0�#����������������ʼּ������ּʼ����������������!�-�:�A�F�S�M�F�>�:�-�!����������׾����	��"�'�)�"��	�����׾Ӿɾʾ׻l�x�����������������y�x�l�a�l�l�l�l�l�l���������Ⱦʾ��������������������������������������������y�v�y������������������(�5�A�C�N�_�d�f�e�Z�N�A�5�.�(�$���!�(�ּ��������������ּ˼ʼʼʼʼּּּּ������ż�������������������������������������#�2�?�=��
�����������������������n�{ŇŊŔśŠŢŠŔœŇŅ�{�o�n�d�g�n�n¢���(�5�8�A�B�F�A�<�5�(�����	������������������������������{�z�{�z�}���¦²¿��������������¿²¦£¦E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�ExE�E��I�H�C�I�O�U�b�g�j�b�a�U�I�I�I�I�I�I�I�I C U ; 2 @ t H 7 0 ( B 1 3 ; = 7 3 X 2 G 8  B S S X N Q < ! � =  & R > P Q f . % 8 T " p Q " ) H B c . 8 1 X .  �  �  �  c  �  i    +  [  8  t    �  i  �  �  �  T  �  �    4  �  v  �  �  *  �  �  �  o  �  ;  ~  c  n  ;  R  �  g  �  �  �  �  i  �  �  '  �  �  6  2  �  
  �  l  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  CE  i  |  �  �  �  t  g  W  F  3      �  �  �  �  �  �  �  r  �  �  �  �                      �  �  �  p  ]  J  V  y  �  �  �  �  �  �  �  w  ^  H  /    �  �  �  e  !    �  �  �  �  �  �  �  �  �  �  z  n  _  N  8    �  �  �  �  �  �  �  �  �  �  �  �  �  |  c  D    �  �  �  i  3  �  �  �  �  �  �  �  p  Z  E  .    �  �  �  �  �  �  t  i  ^  S  i  �  �  ~  s  a  D    �  �  �  ~  V  -    �  �  Z  ,  �  e  �  �  �  �  �  �  p  Q  )  �  �  \  �  R  �  Q  �  :  f  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  �  A  ~  �  �  �  �  �  �  �  �  I  �  ]  �  -  i  6  h  �      !          �  �  �  �  q  <  �  �  +  �  4  �          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  n  i  c  X  J  ;  +      �  �  �  �  �  m  L  (    �  �    ,  C  P  R  G  *    �  �  "  $    �  h  �  a  �  �           �  �  �  �  |  H    �  y  7    �  f  �  �    v  t  p  h  Z  E  )  
  �  �  �  j  =  	  �  �  A  �  �  �  
L    F  *  �    J  T  .  �  �    g  �    
  	  �  '  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  k  `  :  _  �  �  �  �  �  �  |  P  "    �  p  
  �  �    �  �  z  m  `  Q  A  1       �  �  �  �  �  k  E    �  �  �  
  �  �  �  �  �  �  �  r  C    �  �  {  E    �  1  �  �  0  �  p  /  �    d  �  �  �  �  �  �  �  �  �  �  �  p  
  0  ?  7  ,      �  �  �  �  |  S  '  �  �  �  ?  �  �    /    /  k  o  x  �  �  �  �  �  l  8  �  �  �  S  '  �  9  h  �  �  |  t  m  f  _  W  P  I  @  5  *         �   �   �   �  d  `  a  ^  H  /        �  �  �  �  y  N    �  �  _           �  �  �  �  �  f  `  Y  M  /    �  	  �  �  �  �  �  �  �  �  q  M  $  �  �  �  �  W  $  �  f  �  y  �  x   �  I  6       �  �  �  �  `  >    �  �  �  r  8  �  �  D  �    +  A  R  U  K  8       �  �  �  y  ?  �  �  A  �     �  �  �  �    y  t  n  b  Q  A  0               	  
         %      �  �  �  �  �  �  h  K  %  �  �  >  �  �  "  �  �  	  	?  	d  	�  	�  	�  	�  	  	<  �  �  l    �  �  �  W  J  �  �  �  �  �  �  �  �  o  E    �  �  8  �  g  �    �  �  �  �  �  �  �  �  �  z  k  \  M  >  /         �   �   �   �  �  �  �  �  �  v  h  Z  L  =  -      �  �  �  �    c  G  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    i  �  �  �  �     	        �  �  �  �  2  �  y    �  �  �  �  �    v  q  d  V  K  @  4      �  �  ~  Q  (    �  =  �  �  C  h  O  t  d  w  j  q  /  �  �  �  �  �  
�  0  �  2  ?  I  D  9  )    �  �  �  `    �  m  �  g  �  �  �   �  t  b  N  9  $    �  �  �  �  n  D    �  �  x  A    �  �  �  �  �  �  �  �  g  J  /    �  �  �  �  �  �  u  r  �  �      �  �  �  �  �  p  L  ,    �  �  �  �  z  ^  B     �  �  �  �  �  �  �  u  h  W  ?  &    �  �  �  }  M     �   �    R  �  �  �  |  p  [  ;    �  �  -  �  {  	  �  �  7  :  W  D  1      �  �  �  �  �  t  _  J  .    �  �  �  9   �  �  �  �  �  �  �  �  �  �  �  �  �  z  l  ^  O  @  0  !    �  9  h  z  ~  p  P  )  �  �  �  6  �  *  U  
S  	.  �  _  b  �  �  �  �  �  �  �  W  &  �  �  �  K  	  �  |  9  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  5  �  �  }  ?  )  2  6  6  4  3  2  ,  $         �  �  �  �  H    �  .  g  \  N  H  D  9  +      �  �  �  g  
  �  0  �  ,  o  �  �  �  �  �  �  �  �  �  o  D    �  k    �    S  �  �  %    �  �  �  f  )  �  �  �  �  �  j  (  �  �  �  �  ,  �  �  b  R  B  1      �  �  �  �  �  �  �  s  R  %  �  �  �  V