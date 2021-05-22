CDF       
      obs    ?   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�?|�hs      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�6   max       P�(      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       <�o      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?
=p��
   max       @F��\)     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��=p��
    max       @v
=p��     	�  *x   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @Q�           �  4P   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���          �  4�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       ;�`B      �  5�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�"�   max       B4��      �  6�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B4��      �  7�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >{&�   max       C���      �  8�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��   max       C���      �  9�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          S      �  :�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          K      �  ;�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          G      �  <�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�6   max       P�      �  =�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���$tS�   max       ?�/��v�      �  >�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       <e`B      �  ?�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?z�G�   max       @F��\)     	�  @�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���\*    max       @v~=p��
     	�  Jx   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @Q�           �  TP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��           �  T�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�      �  U�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�������   max       ?�-w1��     �  V�         #   %   S   
         "   6   7            I         .            	      	      %                              #   
            1      !                                       1   >      	   +         
O�TEN�biP8Of�nPZ�LO	�fO�E�N5�P.�PG�P\�N!��N�"O(:P�(P/��O�"O��CO>�!NT#O&�tO7c3N03xN̨O� �O�n&Oyk{N
:N���N`{�O(�CN-��OY�O��N�<DO�{N�N3f�N.~[N��P9��No�N�+mN:ǸN��O$�oN��N��O{FO?��NFgrO�^O��;N`�3O�ҴOԧ�P+|�O1RINjAO�1�N.�M�6N���<�o<e`B;�`B;�`B�D�����
�ě��ě��ě���`B�t��t��#�
�T���e`B�e`B�u�u��t����
��1��9X��j������/��`B�C��C��C��\)�t��t����#�
�,1�,1�49X�49X�8Q�8Q�8Q�@��@��@��D���L�ͽY��Y��Y��]/�]/�aG��m�h�m�h�u�y�#��7L��O߽�O߽�hs��{�� Ž�����������������������sz���������zzussssss�������-0/&���HO[hlu����~th[OMJHFHip���������������zhiBHNOS[ehkkih`[OBB@?B
#/HQUWQH</##0740# 6COfu��~lh\O6* %6BO[t�����th[B9'TUY`z������������maTKOW[htuth[OJKKKKKKKK�����

��������������������������������[t��yq^L5)���������
 ���������� $)06BHO[eonh[B)��
#-57684#
�������������������������dhmrt|�������tqhdddd���)-2)����
)6DOZXOB?6)
��������������������.0<IUbnqonbUUIC<60..��������������������).<BN[gt}���tgN5!)]amz���������zma][[]��������������������
#+/7<CA<4/#
����������������������������������������x{���������{xxxxxxxx "%'/28;?AHKKLLH;+" 7BO[ht�������th[OC:7��������������������#)06:<EE</#
`abcmzz|}zxrmjaa````����������������������������������������������������|���������������������������������������������������������������������������������$"	@DLOT[^hktytlhe[OCB@��������������������;<IIUVXWVUNIID><813;9<AILQUY[\\UI=<95549ntx��������������tndnxz|��znadddddddddd?JSfrw����������tgR?nt������������~tmijnvz~���������zrvvvvvv��������������������`lz����������zpfZXY`���+12.)������������������ ���������������������|���������~z||||||||BHUUXXUSHGBBBBBBBBBB^aenz������znja_^^^^�f�d�b�]�[�c�s����������˾Ǿ���������fìâäàÕàìù����������ùìììììì������¿¦¦²¿�����������Y�M�H�I�L�Y�e�p�r�~�������������~�r�e�Y�ɺ������j�o���˺���=�)��	���	�ֺɾ�����|������������������žȾ��������������������������������������������������/�/�"�"�/�;�C�A�<�;�/�/�/�/�/�/�/�/�/�/�;�"�����	��.�G�T�m�y�}�w�s�n�_�U�;�����߾۾���2�;�T�`�o�q�y�{�s�m�T�;��I�<�0�#��� ���#�<�n�{ŃŒŇŀ�m�b�I�x�w�x�z�y���������������x�x�x�x�x�x�x�x�|�v�z  �����ɾʾϾо̾ʾ¾����������������������������l�D�N�s��������a�t�n�H�"�	�������z�s�g�a�P�N�Z�������������������������z�2�"��������������	��/�H�V�_�h�e�T�H�2�û������������ûл����#�&�$�����ܻþʾɾ��������ʾ׾��������������׾��������s�o�g�_�b�g�k�s���������������������������������������������������������������������������������������������������������������������������-�(�$�#�$�&�+�-�:�<�C�F�F�L�F�A�:�/�-�-�Y�L�C�3�&�'�2�6�@�Y�r�����������~�r�e�Y���ʾþžξؾݾ����	��$�.�,��	����ƳƪƘƘƕƚƧ������������������������Ƴ��������������������������������������������������������� �
������
�
�	�����ѿʿĿ��������Ŀѿտݿ�����ݿ׿ѿѻлλû����������������������ûȻѻٻһ��(�"���
���(�5�>�A�J�A�5�(�(�(�(�(�(���������������{�����������������������׽Ľ��������������������Ľн����ݽн�����������������������������������������E�E�E�E�FFF$F1F5F=FJFJFGF@F1F+FFE�E��H�>�;�/�&�.�/�;�H�R�T�a�j�a�U�T�H�H�H�H����ּҼҼּ����������������6�6�*������*�6�C�H�C�6�6�6�6�6�6�6�����#�'�4�@�H�@�;�:�4�'�������Y�M�\����̼���(�*�$������ʼ����Y���	�	��	���"�,�*�"��������������������!�-�+�%�!��������!��������!�#�-�7�2�-�!�!�!�!�!�!ŹŵŭţšŭŹ������������������ŹŹŹŹ�Ϲù����������������ùϹܹ����ܹڹϼY�X�M�@�7�@�B�L�M�Y�f�l�q�l�f�Z�Y�Y�Y�Y�����������������ĽʽнݽݽнĽ½�����������������
��(�4�A�L�I�A�@�4�(�����
�������������������������������ܻܻܻ�������������������I�0�$���������������%�A�S�_�b�_�V�I�{�u�g�\�X�b�n�{ŔŠŭŻ��ŽŹŭŠŔŇ�{�����������������Ŀѿҿѿ̿Ŀ������������#��
����ĿĳęĜĳ�������
��0�B�H�<�#�������������������Ϲ������ܹù�����ùðëéñù��������)�B�I�=�2������ù�l�`�Q�`�l�q�y�����������������������y�l�.�'�!�����!�.�0�:�@�G�K�G�:�.�.�.�.�����������������ĿѿܿտտѿǿĿ��������#���#�/�<�H�I�H�>�<�/�#�#�#�#�#�#�#�#EPEFEOEPE\EiEqEiE_E\EPEPEPEPEPEPEPEPEPEP�����ݾ׾Ӿ׾������������������  9 > / R , 4 ; - * . z b ? T F 6 L   F Q 8 l B - , R q X M { p @ 1 z Z ] � D o J X d i 1 8 A S m ; T : ! { S 8 8 a G = p g    �  �  ~  �    6  ;  X  �  a  �  �  �  J  �  +  
  c  �  �  r  �  K  M  B    �  H    �  �  m     U  �  �  �  a  �  �  W  �  !  j  �  g  �    _    X  n  #  b  �    
  y  �  :  �  2  �:�o;�`B��`B����{�u�ě��t��#�
�y�#��%�e`B��9X��`B��-�,1�49X�u�������������ͽC��P�`��o�ixս��<j��P�]/���ixս�%��C�����Y��@��@��@�����Y������H�9�aG����
��C���o��7L��+��7L��1���㽃o���w��;d�����
���-��l���j��^5��x�B!�B T�B��B�sB��B��B!B%Z�B0�?B��B s�Ba3Bf�B4��B��B��B]�B$�_B�:Be�BZB|B}�B'�B"oEB_�A�PB�B�B*><B�WB)4uA�"�B��B��B��A���B��B[EB)��B-�BB])B"�PB�BIlB ��B&�iB&��B��B�@B
g�B
�wB��B	-B��B0	B|�B7�Bc�B
��BqB#�B!B�B �B@BG%B��B��B3�B%nlB0>�Bc!B �B�FB@�B4��B@B@�BF#B$cB\B��B=�B��B��B'@YB"v�B��A��RB��B@�B*@IB"B)PJA���B>	B��B�A��B��B?�B)�vB,�1B?�B?�B"�fB��BC�B ��B&��B&�sB�B;(B	��B
�{B�RB%B6mBP�B��BE�B&�B
�+BC'B�KAG�eA��FA��L?���@>9YAJ�JA���A�՞AbP�Aa��A�{@��#A�OaAM��A�l�A��A���@��AAS|OA���AI#�A��A��#@s��?��bAY�B�8B9.A�^Az�@�� A��qA�jDA%{MA�J�C���A�f!Ao�A��L@˒6@��[A]6�A	r.@jA�-�>|�}@ڛ~A#�(A5{A�?�@��`B
�A��CAw�rA�n3>{&�A��A��AHaAvT{A6C�ƑAVM�AH�A̕�A�o�?��5@@&�AJ�A�+A��Aa�Ab�aA몋@� MA��AM�A��A���A�^�@���AS��A�~�AH�A���A���@m7?��AY�B4�B3wA��)Az�`@�P6A���A��A#T�A���C���A�RvA�~A��x@��[A �)A]�A�@s< A�J>W��@�#�A"�A6M�A��@�xB
�YA��Awc�A�c�>��A�~rA��A̛Au0
A���C���AU>�         $   &   S            #   7   7            J         .            	      	      %                              #   
            2      !                                       1   ?      	   +                  )      =            /   /   '            K   1   %   %                        !                                             7                                 )         )   %   )                           !                  /   #   #            G   +                                                                           7                                 )         )   #   '                  O���N�biO��rN��O��)O	�fO�uN5�P.�O�-�P�7N!��Ncm�O(:P�P&��O��O�!.O>�!NT#N��AO7c3N03xN̨OE�sO��Oq�N
:N���N`{�O(�CN-��O��Oy��N�<DN�ْN�N3f�N.~[N��P9��No�N�+mN:ǸN��N�ÊN~�N�S-O{FO$�uNFgrO�^O��;N`�3O�ҴO���P٪O��NjAO�1�N.�M�6N���  6  C  K  F  �  k  f  �    +  �  �  K  4  �  �  "  �  �  �  �    �  z  E    N  �  �  L  �  v    �  �  
}  6  U  j      �  
  #    �  �  R  �  �  �  @  #  �     �  	�  �  �  	�  �  c  �<T��<e`B:�o�ě��\)���
��`B�ě��ě������u�t��D���T����1�u��9X���ͼ�t����
���ͼ�9X��j�����o�C��#�
�C��C��\)�t��t��,1�0 Ž,1�8Q�49X�49X�8Q�8Q�8Q�@��@��@��D���}�aG��]/�Y��aG��]/�aG��m�h�m�h�u�����t���\)��O߽�hs��{�� Ž�����������������������sz���������zzussssss�����#&$!�����MOV[hktx�~tjh`[OOMMMz~���������������~{zBHNOS[ehkkih`[OBB@?B	#/AHNSVPH</#		#0740# 6COfu��~lh\O6* )1BO[duy||uh[O3,'%&)]ez�����������me^XY]KOW[htuth[OJKKKKKKKK��

�����������������������������������[gt�{sf[QB5)��������	����������,6BOR^fjijha[OB6)"%, 
!'.00-+(#
����� ��������������������dhmrt|�������tqhdddd��#���������
)6DOZXOB?6)
��������������������.0<IUbnqonbUUIC<60..��������������������&.5BN[gt�~sg[B50)&&`ajmz������zymia_]^`��������������������
#+/7<CA<4/#
����������������������������������������x{���������{xxxxxxxx*/0;@GHIIJIH@;6/&$(*=BGORht������th[KB@=��������������������#(049<?</#`abcmzz|}zxrmjaa````����������������������������������������������������|���������������������������������������������������������������������������������$"	KOV[hhqlha[QOHKKKKKK��������������������<<BIUVXVVUIIHA<:34<<9<AILQUY[\\UI=<95549{������������������{dnxz|��znadddddddddd?JSfrw����������tgR?nt������������~tmijnvz~���������zrvvvvvv��������������������[cnz����������zqh\Z[��).0-)������������	����� ���������������������|���������~z||||||||BHUUXXUSHGBBBBBBBBBB^aenz������znja_^^^^�s�o�j�g�b�b�k������������ľ¾��������sìâäàÕàìù����������ùìììììì��¿¦¦²¿���������
��������e�a�Y�V�U�Y�b�e�r�~������������~�r�e�e�ֺɺ������������ɺֺ���� �������־�����|������������������žȾ��������������������������������������������������/�/�"�"�/�;�C�A�<�;�/�/�/�/�/�/�/�/�/�/�;�"�����	��.�G�T�m�y�}�w�s�n�_�U�;��	�������	�"�.�;�G�T�c�m�j�T�G�;�.�"��0�#����	��#�<�b�n�wń�|�i�b�U�I�<�0�x�w�x�z�y���������������x�x�x�x�x�x�x�x�y�|�����ɾʾϾо̾ʾ¾��������������������������z�L�M�X����������;�e�l�a�H����������s�g�b�R�Q�Z�g�������������������������� ����	��"�/�;�H�M�T�Y�_�b�_�W�H�/������������ûлܻ���������ܻл��ʾɾ��������ʾ׾��������������׾��������s�o�g�_�b�g�k�s���������������������������������������������������������������������������������������������������������������������������-�(�$�#�$�&�+�-�:�<�C�F�F�L�F�A�:�/�-�-�L�H�@�3�7�@�L�Y�e�r�y�~�������~�r�e�Y�L�	���Ͼ;վ޾����	���$�)�&�"���	ƳƲƧơƠƧƳ������������������������Ƴ��������������������������������������������������������� �
������
�
�	�����ѿʿĿ��������Ŀѿտݿ�����ݿ׿ѿѻлλû����������������������ûȻѻٻһ��(�"���
���(�5�>�A�J�A�5�(�(�(�(�(�(�����������������������������������������Ľ��������������������Ľнܽ��ݽԽн�����������������������������������������E�E�E�E�FFF$F1F:F=FBFEF>F1F)F$FFE�E��H�>�;�/�&�.�/�;�H�R�T�a�j�a�U�T�H�H�H�H����ּҼҼּ����������������6�6�*������*�6�C�H�C�6�6�6�6�6�6�6�����#�'�4�@�H�@�;�:�4�'�������Y�M�\����̼���(�*�$������ʼ����Y���	�	��	���"�,�*�"��������������������!�-�+�%�!��������!��������!�#�-�7�2�-�!�!�!�!�!�!ŹŵŭţšŭŹ������������������ŹŹŹŹ�ù����������ùϹ׹ܹ�ܹ۹ϹùùùùùüY�M�M�E�M�N�Y�f�j�o�j�f�Y�Y�Y�Y�Y�Y�Y�Y�����������������ĽƽнҽнĽ���������������������
��(�4�A�L�I�A�@�4�(��������������������������������������ܻܻܻ�������������������I�0�$���������������%�A�S�_�b�_�V�I�{�u�g�\�X�b�n�{ŔŠŭŻ��ŽŹŭŠŔŇ�{�����������������Ŀѿҿѿ̿Ŀ������������#��
����ĿĳęĜĳ�������
��0�B�H�<�#�������������������Ϲ��������ܹù���õîëôü��������#�)�9�<�7�-������õ�y�l�`�U�`�e�l�s�y���������������������y�.�'�!�����!�.�0�:�@�G�K�G�:�.�.�.�.�����������������ĿѿܿտտѿǿĿ��������#���#�/�<�H�I�H�>�<�/�#�#�#�#�#�#�#�#EPEFEOEPE\EiEqEiE_E\EPEPEPEPEPEPEPEPEPEP�����ݾ׾Ӿ׾������������������   9 > '  , 4 ; -  & z * ? W B  L   8 Q 8 l 6 - 1 R q X M { T ? 1 x Z ] � D o J X d i ) = : S ^ ; T : ! { P 0 9 a G = p g    H  �  �    S  6    X  �  �  H  �  w  J  9    @  �  �  �  �  �  K  M  �  �  S  H    �  �  m  �    �  �  �  a  �  �  W  �  !  j  �  �  �  �  _  �  X  n  #  b  �  �  �  X  �  :  �  2  �  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  (  /  4  6  5  1  )      �  �  �  �  �  �  �  y  _  K  A  C  3  $      �  �  �  �  �  ~  e  B     �  �  v  3   �   �  �  "  A  K  H  <  $    �  �  �  Z  )  �  �  r    �  �   �  �  �  �    $  8  A  D  :  %    �  �  k  "  �  9  �  �  �  q  �  F  �    P  w  �  �  |  f  M  (  �  �  D  �      �  k  j  j  g  c  \  S  G  8  (      �  �  �  �  �  �  x  l  b  f  f  ]  K  5    �  �  �  �  �  �  `  /  �  �  z  C    �  �  �  �  �  �  �  y  q  i  a  Y  Q  I  >  3  (          
  �  �  �  �  �  i  O  0    �  �  �  w  K    �  W   �  �  �  �  �  
    )  *      �  �  �  }  I  �  �  �  �   �  �  �  �  �  �  p  I    �  �  �  }  K    �  u  �  ,  ?  _  �  �  �  �  �  �  }  s  g  W  H  8  $    �  �  �  �  �  n        C  D  ;  .  !    �  �  �  �  x  [  8    �  J  �  4  ,  (  "        �  �  �  �  �  �  o  G    �  �  3   �  �  �  �  �  �  �  �  �  ]    �  L  �  �  �  G  �  T  �  �  �  �  �  �  �  �  �  �  u  e  q  g  [  P  g  d  :  �  �  T  �  �      "       �  �  �  �  Z  )  �  �  �  5  �  h   �    h  �  �  �  �  �  �  �  �  �  �  k  M    �  7  �    .  �  �  �  �  �  {  k  ^  T  I  :  %    �  �  �  �  S  %  �  �  �  �  �  y  e  Q  <  &      �  �  �  �  �  �  �  h  N  !  b  z  �  �  �  ~  y  r  e  W  A  !  �  �  �  :  �  �  �    s  g  Z  K  <  +      �  �  �  �  �  {  ]  A  1  7  <  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  l  a  W  M  B  z  p  e  Z  N  B  2  #       �  �  �  �  �  �  q  a  Y  R    )  6  ?  D  B  >  5  &    �  �  �  l  0  �  �  [      �  �        �  �  �  �  o  ;    �  �  �  X    �        9  E  J  M  N  H  0    �  �  y  6  �  �  w    �  ,  �  �  u  j  _  S  H  =  1  &        �  �  �  �  �  �  �  �  �  x  g  P  6      �  �  �  �  f  7    �  �  A  �  �  V  L  C  :  1  (           �   �   �   �   �   �   �   �   �   �   }  �  �  �  �  �  �    o  ]  H  2    �  �  �  y  E    �  �  v  o  i  c  ]  W  P  J  D  >  7  0  )  "           �   �  �  �            �  �  �  }  S  #  �  �  h    �  U  �  n  z  �  �  z  j  W  @  %    �  �  �  j  2  �  �  )  �  w  �  �  �  �  �  �  �  ]  /  �  �  �  >  �  �  U  �  �  =  e  
D  
z  
|  
f  
8  
  	�  	�  	V  	  �  �  m  ?  �  �      ^  �  6  %      �  �  �  �  }  X  ,  �  �  �  o  =    �  �  W  U  I  >  2  '      
        �  �  �  �  �  �  w  Y  ;  j  Z  K  ;  +      �  �  �  �  �  �  �  �  �  �  x  k  _    v  m  d  [  R  I  @  7  .  %           �   �   �   �   �    �  �  �  �  r  9  �  �  �  v  8  �  x    �  '  �  �  V  �  �  �  �  �  �  �  s  ]  G  .    �  �  �  �  \  ,   �   �  
  	�  	�  	�  	r  	:  �  �  s  $  �  r    �  
        �   �  #  !                     �  �  �  �  �  i  O  4      �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  ^  L  :  (  2  k  �  �  �  �  �  �  �  �  �  �  �  T    �  �  F  �    K  L  o  �  �  }  u  i  Y  D  *  	  �  �  �  s  C    �  �  Q  R  P  H  A  ;  0  %    �  �  �  �  {  S  )  �  �  �  V  �  �  �  �  �  �  �  �  i  J  )    �  �  i  %  �  �  3   �  �  �  �  �  �  z  b  H  .    �  �  �  �  �  d  6    �    �  �  �  �  h  C    �  �  �  ~  R  %  �  �  �  ^    �  ~  @    �  �  �  c  $  �  �  �  �  �  �  @  /  B  �  f  �   �  #      �  �  �  �  �  �    e  E    �  �  q  +  �  n    �  �  �  �  �  �  �  s  c  S  C  2  !       �  �  �  �  �           �  �  �  �  �  �  �  �  x  W  0  �  �  �  _  �  �  �  �  �  �  �  t  @     �  S  �  y    �    u  �  �  �  	�  	�  	�  	�  	�  	�  	=  �  x    �  $  �  6  �  �  <  �  �  X  �  �  �  �  �  �  �  �  �  �    j  R  5    �  �  �  a    �  �  �  �    !    �  �  �  �  �  f  K  /    �  �  �  �  	�  	i  	P  	G  	+  �  �  �  \    �  �  #  �  ,  �    �    
  �  �  �  s  _  K  4      �  �  �  �  �  �  }  h  Q  :  "  c  K  2      �  �  �  �  �  �  �  �  �  ~  _  >    �  �  �  �  �  �  �  k  F    �  �  �  �  �  �  m  X  J  >  t  �