CDF       
      obs    =   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�t�j~��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P�r      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �'�   max       >��      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�
=p��   max       @E~�Q�     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @vi�����     	�  *   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @Q            |  3�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�,        max       @��@          �  4   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �\)   max       >k�      �  5   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��'   max       B+p�      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��+   max       B+:k      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =��=   max       C��      �  7�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =49�   max       C�i      �  8�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  9�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          M      �  :�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          A      �  ;�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P�o      �  <�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�1���o   max       ?��|����      �  =�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �'�   max       >��      �  >�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�
=p��   max       @E~�Q�     	�  ?�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @vi�����     	�  I   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @Q            |  R�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�,        max       @���          �  S   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Dg   max         Dg      �  T   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�4�J�   max       ?��c�A \       T�               �            a                                       	   
   %            '         
   ;   9      <                  	   
         9   	   	      =                  �   (         L      2N6��NMiN51=O �|P��Nr�N$zN��P�rN���OW�O��{N��OH�O.t�Nf�Nz��N��O(�,O���NҀ�N�H�N�{mP�'O,�HN���O�jO|1�OBŜN*�N��PA��PA�DO\d'PG��N)HOG��O�f�OCXCN$0^N�.�N� M���O`��O��-N��AN��WN��O���N��VN+iO ��M�z�N���O���O���N�G�N�7QO���N�f�Ob��'\)����u�t��t��t���o��o��o��o;�`B<o<#�
<#�
<�o<�o<�o<�C�<�t�<�t�<���<�9X<�j<ě�<ě�<���<���<���<�/<�`B<�h<�h<�h<��<��<��=t�=��=��=,1=49X=@�=P�`=Y�=e`B=ix�=q��=�7L=��=���=�{=� �=�9X=�E�=���=���=�S�>O�>�+>����������������������snrt�����tssssssssss-036ABFKB6----------@BNT[gtu������tgWNB@���)Bg�����gB)���������������������������������������������������������������������)5Md`N)������)357BNV[bd\[NB85))));4017<HPU\_`\UH<;;;;�����
#5<=:0#���������������������������������

�������������������������RU[bgrtwtga[RRRRRRRR��������������������`^[]acnnnvz�znda``OLKMUYagnz|��~znaUOO������
����)&*/<CHRU`USH<4/))))�����������������������"�������������
����������)6BO[[Y[][OLB=)����������������������������������������������������������
#/1<?EKLH</#
u}����������uuuuuuuu��������������������������&0:2)�����FIUaz�����������naNF��������
������������������������|�������������������������������������
"+-.*#
����"/:;@?;91/'"./2<HIMH</..........�����	

����������������������sqtt�����tssssssssss�����!*++) �����
#/<HSYYSH</��97BOX[dhrh[OEB999999+66;>BO[`d[OJB66++++��������������������$14:3/#
������#$����

�����������������������������������������������IBIOUbdb`UIIIIIIIIII-#$,/;@HJHD;//------���������

������gbz��������������zng��������������������``amoyz}}zzomaZWW[``zx~����������������z������������������������������

�����ż���������������z���������������������Óàì÷õìàÓÉÓÓÓÓÓÓÓÓÓÓÓ����!�������������������������������������������������������������)�B�|ąĆ�|�k�O�6�����������ìá�������
���������������������������������������������������������������čďĚěĚĖĚğĚčĊā�t�n�k�tāĈčč�0�I�{Š������ť�w�U�������ĿħĦ���
�0�t�t�g�g�b�f�g�h�t�t�t�t���������ʾ;ξʾɾ�����������������������(�,�&�2�4�3�&������۽ڽ�������#�/�<�A�D�@�<�9�3�/�)�#�!����#�#�#�#����)�+�.�6�9�9�6�)������������������#�'�.�'�"������ܹϹǹϹѹܹ��ݿ������������ݿٿݿݿݿݿݿݿݿ�������� ������������������������������������������������������m�z���������������������z�m�j�c�c�c�m�m�<�H�a�n�t�xÀÂ�z�n�a�H�<�/�#����#�<���������� ����������������������¿����¿²®¦¦²µ¿¿¿¿�S�_�l�x�z�~�x�l�k�_�S�R�M�H�S�S�S�S�S�S�������������������������g�I�L�T�Z���������ʼּؼڼڼۼ�׼ּʼ������������������r�����������������x�r�o�m�r�r�r�r�r�r���� ���������������ý�����������ù����������ùìÓÆ�z�q�t�zÁÇÓàìù�T�`�m�t�t�o�m�e�`�T�G�;�6�8�6�9�?�G�J�T���������ùʹù����������������������������!�#�(�"�!��������������������Ŀѿ����ݿѿ��y�m�T�O�F�L�T�m�����A�Z�p�{�������������g�Z�N�(��
���(�A����������������	���������������������s���������׾��*���׾����\�Q�N�T�^�s�������������������������������%�'�2�8�=�4�'�����������)�4�9�:�5�<�5�)�������������������/�;�H�T�a�d�l�p�m�m�a�T�@�;�/�*�"��"�/¤�������ʾϾ׾���׾־ʾ����������������;�G�L�Q�S�G�G�;�.�-�"����"�"�.�3�;�;�f�s�������s�f�^�b�f�f�f�f�f�f�f�f�f�fƧƳ���������������������ƳơƧƬơƞƧ�	��/�;�A�J�N�N�G�;�/�"��	����������	�z�����������z�u�m�i�e�h�m�o�z�z�z�z�z�z�����������źú���������������������������!�-�.�:�F�H�S�Y�S�F�:�-�&�!������#�
����������ĶĵķĿ���������
��%�0�#��������������������������������������ҽ�ݽнĽ����������Ľнݽ���������m�z���������������z�m�a�Z�]�_�a�b�g�m�m��������������������������Ŀ����������������������ĿĿĿĿĿĿĿĿD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DzD�D�D����
�%�-�3�5�3�.�!�����������޺��s�����������������������������s�s�q�s�s����������������ŹŭŬŠŞŘŠŭŹ�����Ӽ����ʼּ������ּʼ�����������������¿��������������¿²«¦¢¦²³¿¿¿¿E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E{ExExE� F L @ t H N 6 < B > # S d # _ l a :   Y D z . : J 0 F ; K s 0 H 3 o o Y W 1 9 2 K = O ^ 2 h z t ? 5 � J r * ) g T J 0 < K    I  X  P  �  <  <  8  �  �  (  &  e  M  �  �  r  �  �  e  M  �  �  �  s  �  �      �  �  �  �  A  I    g  �  r  �  H    ,      �  �  �  �  �  �  �  p  =  �  (  �  �  .  �  �  ӽ\)��h�u%   >hs��`B��o;o=�j<t�<49X<���<ě�=�w<��<�t�<��
<�j=\)=49X=��<�/<��=u=H�9=o=<j=�o=0 �=��=��=�9X=�{=P�`=�^5=��=ix�=y�#=Y�=H�9=L��=Y�=H�9=�o=�G�=��=��=�\)>o=�Q�=���=�;d=�Q�=ȴ9>k�>V=�G�>J>Z�>#�
>M��B)ÍB
eB��B	�rB�ZB�AB"�ZBBMzB	�BH�B$��B��B��B��B	B BõBs�Bq
B�B3BB�B��B#%YB�XB!�B�eB��B -B�B�(B�.Bj�B,B!�B��A��'B̫B$C�B"7�B�^B��B�KB�{B��B*�fB(�B�_B+p�B҄B'3oA�	�BȧB��B�A�cB�Bm6Ba�B)�/B	��B�B	@xB�uB��B"��B�B��B	ZBUB%g+B 9B��B�B��B:B� B� Ba�B�|B@%B�pB��BɣB#2HB�B"5XB�6B>�B 2PB bBnB7VB?�B@~B!�=BA�A��+B��B$>�B"<�B\&B�'B=B�BuYB*��B@ B�B+:kBB�B&��A�gB��B�zB=�A���B;�Bs�B�@�XA˝r?e�2A��@AՈ*A��@�D�A���A���A���AL�A1�A�A�ӭ??�<A~P�A�)�A���A�_A��A��A��@��XA���@��@�{JA�~�A�v	Afx�=��=@b��Aq<BA���A�8�AM!@�@��/A�"1A��1A��bAN��Aa�aAB��ByHA�>1A�::@#Ɩ@w^NA���A���A(�[A���@K��A�C��@Z��A��
A��G@���A�s|C��@��Aˀ�?U�A�~�AԘ�A��-@��A�}�A��VA�,AK��A/ĈA��A�t�?OKPA}�A�}�A���A�rVA�{�Aҍ!A�Ŗ@��A�z�@�"�@��A�{�Aˁ�Ag�=49�@dAq�A��xA�e[AM�@��@��A��}A�W~A���AM�KAa �AB�UB��A��+A�z%@$b�@s��A�|�A�y;A(M)A�y5@K�3A�|KC���@c�iA�eA��O@��A�-"C�i               �            a                                       	   
   &      	      (            <   9      =                  	   
         :   
   	      >                  �   )         L      2               I            M         !                                    '                        /   -      7                                                               %                              5            A                                                                        -      1                                                                              N6��NMiN51=N��PN�_Nr�N$zN��P�oN�6^N���Oc��N:�N�N�O.+Nf�Nz��NZ�O(�,O|�EN�'N�H�N���O��Nk��N!��N�1XO/�OBŜN*�N��O���P(
xO>bQP�wN)HO�LO��sOCXCN$0^N�.�N� M���OH��O��eN��ANT��N��OV��N�gHN+iO ��M�z�N���O��O�f�N�G�N�7QO���N�f�Ob�  �  W  /     �  $  �  �  /  �  �  �  �  7  I  J  �  *  h  P  �  W    �  �  s  _  �  �  �  
  �  �  �  �  z  W    �  X  ^  �    �  b  >  X  �  	�  �  p  �  G  J  �    �  �    �  
Ƚ'\)����T��=D���t��t���o<D��%   �D��<t�<e`B<���<D��<�o<�o<�t�<�C�<��
<�9X<���<�j<��=\)<�/=C�=+<���<�/<�`B=Y�=\)<��=�w<��=\)=�P=��=��=,1=49X=@�=T��=}�=e`B=m�h=q��=���=���=���=�{=� �=�9X>�=��=���=�S�>O�>�+>����������������������snrt�����tssssssssss-036ABFKB6----------MNY[gpt�������ytg[NM���)5Yktwj[I)��������������������������������������������������������������������4C[[N5�����68?BIN[\_[QNKB666666<513;<HJU[^`[UH<<<<<���
#(04:<90#
�����������������������������������������������������������RU[bgrtwtga[RRRRRRRR��������������������^`anz}{|znga^^^^^^^^OLKMUYagnz|��~znaUOO������

	����,*/5<=HNUZULH<8/,,,,���������������������� !��������������������������'$)6BMIDB76)''''''''������������������������������������������������������������
#/1<?EKLH</#
u}����������uuuuuuuu��������������������������������HLUaz���������znaQHH���������	���������������
��������|������������������������������������
!*,,)#
�����"/:;@?;91/'"./2<HIMH</..........�����	

����������������������sqtt�����tssssssssss����  )**(���#/<HNUUNH</#
�97BOX[dhrh[OEB999999:;>BO[^a[OHB::::::::������������������������
#+.000/$
������

�������������������������������������������������IBIOUbdb`UIIIIIIIIII-#$,/;@HJHD;//------���������	


������z��������������������������������������``amoyz}}zzomaZWW[``zx~����������������z������������������������������

�����ż���������������z���������������������Óàì÷õìàÓÉÓÓÓÓÓÓÓÓÓÓÓ����!��������������������������������������������������������������)�O�g�q�u�h�[�O�B�6�,����������������
���������������������������������������������������������������čďĚěĚĖĚğĚčĊā�t�n�k�tāĈčč��0�I�U�nŠťŝ�b�I��
������ĿĹĿ����t�t�n�g�f�g�j�t�t�t�t�t�t���������ʾ˾˾ʾǾ�����������������������!�&�0�/�"��������ܽܽ��������/�7�<�?�<�<�/�#�"�#�#�(�/�/�/�/�/�/�/�/����)�+�2�0�)���������������������'�+�'�������ܹϹ͹ϹԹܹ����ݿ������������ݿٿݿݿݿݿݿݿݿ�������� ���������������������������������������������������������m�z���������������������z�m�j�c�c�c�m�m�<�H�a�n�r�v�À�z�n�a�H�<�/�#����#�<��������
�������������������������¿����¿²®¦¦²µ¿¿¿¿�S�_�l�x�y�z�x�l�i�_�U�S�O�K�S�S�S�S�S�S�������������������������q�g�X�U�]�o�������ʼμҼӼʼ�������������������������������������������������{����������������������������������������������àìù��������ùìàÓÇÁ�z�z�ÇÇÓà�T�`�m�t�t�o�m�e�`�T�G�;�6�8�6�9�?�G�J�T���������ùʹù����������������������������!�#�(�"�!��������������������������ĿɿͿ¿������y�m�a�[�`�k�m�����N�Z�j�v�����������g�Z�A�(��
�
��(�A�N������������������	�������������������s��������������׾������}�j�a�^�a�f�s������������������������������"�'�.�3�'��������������)�1�5�7�4�8�5�)�������������������)�/�;�H�T�a�d�l�p�m�m�a�T�@�;�/�*�"��"�/¤�������ʾϾ׾���׾־ʾ����������������;�G�L�Q�S�G�G�;�.�-�"����"�"�.�3�;�;�f�s�������s�f�^�b�f�f�f�f�f�f�f�f�f�fƧƳ���������������������ƳƬưƧƣơƧ��/�;�E�H�I�A�6�/�"��	�����������	��z�����������z�u�m�i�e�h�m�o�z�z�z�z�z�z�������ĺ���������������������������������!�-�.�:�F�H�S�Y�S�F�:�-�&�!���������������
����
�����������������������������������������������������������ҽ�ݽнĽ����������Ľнݽ���������m�z���������������z�m�a�Z�]�_�a�b�g�m�m��������������������������Ŀ����������������������ĿĿĿĿĿĿĿĿD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D������!�-�1�1�3�2�,�!������������s�����������������������������s�s�q�s�s����������������ŹŭŬŠŞŘŠŭŹ�����Ӽ����ʼּ������ּʼ�����������������¿��������������¿²«¦¢¦²³¿¿¿¿E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E{ExExE� F L @ i [ N 6 < = 2 " M O  [ l a (   \ = z + 2 ? K 8 - K s 0 = - e k Y M 0 9 2 K = O ` / h U t 7 * � J r *  O T J 0 < K    I  X  P  *    <  8  �  O  �    �  e    |  r  �  l  e  1  �  �  �  �  �  N  �  w  �  �  �  q  �  �  B  g  P  V  �  H    ,    �  I  �  �  �  �  �  �  p  =  �  G  �  �  .  �  �  �  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  Dg  �  �  �  �  �  �  �  �  �  w  f  T  B  .       �   �   �   l  W  F  5  $    �  �  �  �  �  ]    �  w  O  &   �   �   �   w  /  )  #               �  �           -  ;  I  W  f  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  f  @    �  �  
�  C  �  �  3  �  �  �  �  �  �  B  �  1  
�  	�  �  t  �  !  $  "  !                  �  �  �  �  �  �  |  g  R  �  �  �  �  �  �  �  �  �  �    k  V  A  +       �  �  �  �  �  �  �  {  p  f  W  E  3  #      �  �  �  �  l    �  �    #  /  (    �  �  �  A    �  �  j    �  4  ~  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  p  j  �  �  �  �  �  z  p  c  S  B  .    �  �  �  u  8  �    �  �  �  �  �  �  �  �  �  �  �  }  a  ?    �  �  m  0  �  �  �  �    ;  ^    �  �  �  t  P  &  �  �  {  2  �  �  2  �  �  �  �    +  5  7  1  '      �  �  �  I  �  ~  �  m    @  H  G  9      �  �  �  �  q  H    �  �  �  X  �  �  A  J  G  D  A  >  ;  8  6  3  0  ,  (  #            	    �  �  �  �  �  �  �  {  n  a  Y  U  R  N  K  E  ?  9  2  ,  �  �        '  )  '  &  !      
  �  �  �  �  �  �  �  h  e  ^  R  E  6  $    �  �  �  �  W    �  U  �  �  l  0  G  N  M  C  0       �  �  �  /      �  �  �  Y  	  {  �  y  �  �  �  �  �  �  �  k  L  "  �  �  w  4  �  �  e  8    W  Q  J  U  i  x  w  u  k  _  O  <  (    �  �  �  K  �  �            �  �  �  �  �  �  �  z  ^  D  *    �  D  �    \  x  �  �  �  �  m  L  "  �  �  O  �  �  _    �    �    ^  �  �  �  �  �  �  �  �  �  �  �  �  ~  t  d  N  '  �  o  p  r  r  r  r  s  s  v  z  y  s  k  V  @  .    $  S  �  Q  =  &  	  �  �  �  H  _  Z  K  4    �  �  �  |  H  �  �  g  �  �  �  �  �  �  �  T    �  �  &  �  �  �  _  -  �  �  �  �  �  �  �  �  �  �  i  D    �  �  g  *  �  �  �  s  �  �  �  �  �    ]  m  n  v  �  �  L  �  �  3  n  t  9  B    
  �  �  �  �  �  �  �  �  {  i  V  D  4     
  �  �  �  �  O  �  �    ;  d  �  �  �  �  v  J  (  �  �  Q  �  K  �     �  �  �  �  �  �  q  G    �  �  t  3  �  y  �  d  �  c  �  x    �  i  G  !  �  �  �  _  *  �  �  �  �  �  z  E  �  f  R  �  �  �  �  �  �  �  �  �  �  w  F  �  ]  �  �  8  9  6  z  |  ~  }  w  q  q  u  z  e  M  5       �  �  �  �  r  T    /  @  W  S  D  *       �  �  �  x  <  �  �  S  �  .  �  �    �  �  �  �  �  �  �  �  g  C    �  �  �  �  8  �  X  �  �  �  �  �  n  S  2    �  �  �  b  3     �  �  =  �  �  X  \  _  `  _  \  Z  W  O  G  =  2  %        �  �  �  �  ^  Z  U  O  G  >  4  +  !      �  �  �  }  G    �  �  G  �  �  �  �  �  �  �  �  s  ^  F  *    �  �  �  E     �   �            �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  �  �  �  �  �  �  �  �  o  ^  e  L  &  �  �  �  �  s  �   �  4  T  _  b  _  Y  N  A  -    �  �  R  �  Y  �  #  d    �  >  #    �  �  �  �  �  }  e  N  6      �  �  >  �  F   �    .  M  V  S  O  G  ?  /      �  �  �  �  �  �  l  F     �  �  �  �  �  �  �  �  h  K  ,    �  �  �  `  ,  �  �  �  	8  	Y  	|  	�  	�  	�  	�  	�  	S  	  �  ~  !  �  N  �  �  �  P  ~  �  ~  ~  �    p  _  H  .    �  �  �  `    �  Z  �  4  	  p  b  S  E  7  )        �  �  �  �  �  �  �  �  �  �  �  �  \  %  �  �  �  A  �  �  j    �    �  f  z  j    �  C  G  .    �  �  �  �  s  J  "    �  �  �  �  �  �  �  �  �  J  1    �  �  �  �  �  [  0    �  �  w  B    �  �  x  e  r  �  !  q  �  �  �  �  ]    �  2  �  �  �  @  �  
�  �    �  �    �  �  w  :  �  �  R  �  �  ]    �  X  �  0  D   T  �  �  �  m  I  "  �  �  �  �  _  8    �  �      $  $  #  �  y  l  c  L  -      �  �  �  �  g  >    �  �  �  Q      �  y  Y  /  �  �  e    
�  
I  	�  	�  	j  �  `  x  Z    �  �  �  ]  1    �  �  �  �  l  H  '    �  �  �  d  8     �  
�  
�  
�  
k  
;  
  	�  	�  	�  	y  	C  �  �  M  �  J  l  �  g  �