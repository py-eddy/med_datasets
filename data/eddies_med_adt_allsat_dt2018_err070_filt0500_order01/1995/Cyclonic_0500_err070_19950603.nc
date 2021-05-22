CDF       
      obs    Q   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��j~��#     D  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P��P     D  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��v�   max       <�     D   4   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>\(�\   max       @FxQ��     �  !x   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �У�
=p    max       @vq\(�     �  .    effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @Q            �  :�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�?        max       @�P�         D  ;l   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �o   max       <�j     D  <�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��o   max       B1e     D  =�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�   max       B0İ     D  ?8   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��B   max       C�LT     D  @|   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�U�   max       C�VH     D  A�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          Q     D  C   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?     D  DH   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =     D  E�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P��     D  F�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��a��e�   max       ?�6��C-     D  H   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��v�   max       <�`B     D  IX   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>^�Q�   max       @FxQ��     �  J�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�     max       @vq\(�     �  WD   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @Q            �  c�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�?        max       @���         D  d�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @�   max         @�     D  e�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?v�����   max       ?�5?|�i     P  g            
   .         Q   &      *   )   +                           *                        '                                                         
   >   -                     $         ,      2      6      $            M                           !   N�&N�eKO�>Np�iP��POv�O�>bO��O�?N|�eO��PTanO�f6Od�N�� O�kxP,%N I�OQ1O��O�P��N���P��O�="Oe�rNHN���O!O���N�b�N�R�M��O�#�O��NN���O�?6O#(�O���N
��Oִ�N�AO ӰN�P�N=iO"�OB92N��O�"�O�y�O=�N�I�O�`�M�ojN���N���O<��Oh�EO;R�O�L~N��O�4�N�E�O�:�N�CTO�]�O9!ODV7N��XO�
N��O��ON?�N���O)��N�9�N0x{O�,OòNJ�-<�<�`B<�9X<�o<e`B<#�
;ě�;��
;D���o���
���
�ě���`B��`B�t��49X�49X�D���D���T���e`B�u��o��C���t���t����㼛�㼣�
���
��1��1��1��j��j��j�ě��ě����ͼ��ͼ��ͼ��ͼ�/��/��`B��`B��h��h��h�����t���P�#�
�0 Ž49X�49X�8Q�<j�<j�@��@��D���H�9�H�9�H�9�L�ͽL�ͽL�ͽP�`�P�`�P�`�ixսy�#��o��C���C���O߽��T��v�)56:8660)) ����������������������������������������26BEOS[OB;61222222220BOh�����h\B

�����������������������(*)
�������������&'%!%%������
/<Uai_US</#
��������������������EQT_bmpv���zmTH;302Exz����������������~x�
#).4763/
�������������
#%/7<@<6#������������������������������ ����������Tamz�������zla][YMJT46BDKOPOB@>644444444SVamyzzwtqooma[VQPQS�����������������{|�
#*/;<><4/#
 ��0@Lc{������{U<0����������������������x���������������}|wxox���������������yno���� 	
���������167BBDOTTOOMB6111111���(!����������������������~}U]amz������omaZTONPU��������������������'),36?BHOPONFB76*)%'��
���������(.6COhu����k\UB6* !(dgjt������������tkgd����������������������������������������+5N[g����tichg[NB5(+[`git~������tpjgbb[[������������������������������������������������������������������������������������������������������������������������#.+&#,/4<HUasz����zaUH</,vz���������������zvv������������������������������������������).0265)�����46BLO[htuvsih[OB6,*4st����������tokissss��������������������()5BCDB53)((((((((((���

�����������|���������~{{z||||||	#(/<@>:951*#	DO[hrt�������tg^[ODD#(),7=<<64/$	<DKO[ho{����zth[B<8<������������������������������
������35:BGNSZ[`[NB5523333`t��������������tg[`�������������������������!&$��������������������������#0<INSTSQI<40#<<DGILOSTSOID<;779<<)6BOU^_^ZOB6)%��������������������������
���������mnz�����zqnnmnonmmmmw�������������wwwwww����������znca`aanz�����������������������������������������(/9<HQUNH<9/)#Taz����������}wnmjUT���������������ֺҺкֺ׺������
��������ֺֺֺ�������������
������
���������������~�u¦²¶¹µ²¦�U�I�P�U�[�a�n�v�t�n�f�a�U�U�U�U�U�U�U�U�;�	�����"�;�G�y���Ŀοпǿ������m�;����������������������������������������ÇÁ�~Ëàìù��������������������ìàÇ�f�W�L�6�1�@�M�f�������������������r�f�(�����!�)�B�H�O�v�w�}�|�t�h�[�O�B�(�����������Ŀʿѿܿݿ޿ݿҿѿпĿ��������t�h�b�[�[�e�h�tčĦĳĽĽĹĶĳĦĚč�t����ĿĸĲĸĿ������+�I�b�i�f�Y�U�<����I�@�<�D�N�Z�s�������������������s�g�Z�I������¿²§¦¦²�����������������������Z�Y�V�Z�]�g�s���������������s�g�Z�Z�Z�Z�m�f�g�o�y�����������������������������m������������������"�-�4�H�'�	�����������_�]�\�_�l�q�x�����x�l�b�_�_�_�_�_�_�_�_�N�A�+��(�5�A�N�g�s�������������s�g�Z�N����������������������#�,�1�4�*�����ݽԽнͽͽнԽݽݽ����������������������c�Z�N�A�+�0�N�g�������������������o�d�b�Z�_�b�o�{Ǉǈǎǌǈ�{�o�o�o�o�o�o�������{�`�S�T�m�����Ŀѿ��
������Ŀ��"�������	��"�/�;�H�S�]�]�T�H�/�"�	�����������	��"�/�;�M�R�H�D�;�/�"��	�������������������ûûû����������������������������������������ʾ;ʾ���������������������������������������������ƚƁ�w�g�g�uƃƚƧƳ����������������ƳƚàÝÓËÇ�z�z�r�zÇÓÖàäììøìàà�H�A�;�/�*�"��"�/�;�@�H�T�W�a�c�a�a�T�H�G�F�;�9�.�-�.�;�?�G�K�H�G�G�G�G�G�G�G�G����پ̾Ⱦվվؾ�������	���%�"����	�������������	����"�$�'����	����޾�����������������������������������������������������������������o�k�x�t�z���������������������������������ÿѿӿݿ����������ݿѿƿ��ֿͿĿ��������������ѿ������������ֿy�s�m�h�m�y���������y�y�y�y�y�y�y�y�y�y��������������)�B�]�e�V�N�A�5�)���àÜÓÓÓÖÛàìù����������ùùìàà�����(�.�2�4�5�4�2�(�"����
������~�x�l�j�_�]�Z�_�l�x�|�����������������B�<�5�,�5�B�N�W�[�_�[�N�B�B�B�B�B�B�B�B�U�J�H�F�E�B�H�H�J�P�U�a�j�j�f�g�c�`�Z�UìåàÞÔÌÇÅÇÓàìó����������ùì�$����������$�(�0�2�6�8�6�1�0�$�$�4�"������4�@�Y�f��������������Y�4�O�B�4�)�%��.�B�O�[āĄČĆā�t�p�h�[�O�������������������������������������ŇŇņńŇŎŔŠŢŭűŴűŭŠŔŇŇŇŇ�������#�0�K�W�]�d�r�y�b�U�I�<�0�������������������������������������������|�{�{��������������������������������¿´·¿��������������������¿¿¿¿¿¿E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eٽ������������(�4�A�G�C�4�(�"��������ݽнʽнݽ����	���(�4�9�4�(���ù��������������ùϹܹ����������ܹûܻڻػջܻ޻����ܻܻܻܻܻܻܻܻܻܻû����x�_�V�U�\�h�x���������ѻջ׻лʻ��*�"����$�*�6�:�C�G�M�K�C�C�6�*�*�*�*�/��������
��#�/�<�H�[�_�_�U�Q�I�/�t�n�h�a�e�h�p�tāăččėĚĜĚčā�t�t�ʼ������żѼ�����!�*�*�"������ּ�����
���(�5�A�N�Z�g�a�Z�Q�A�5�(�����������������������Ľн۽ӽнϽǽĽ���������������(�4�?�A�C�A�4�(����g�b�[�^�g�v�������������������������s�g�~�{�r�m�e�`�d�e�r�}�~�����������������~���������������ɺ����$�!����ֺɺ����������������ʾԾ׾׾׾ʾƾ�����������������üú�����������������������������ҿ����������Ŀѿֿٿڿܿڿѿ̿Ŀ��������������� �!�.�6�:�1�.�#�!�������:�9�:�D�G�S�`�l�`�U�S�G�:�:�:�:�:�:�:�:D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ùƹϹ׹����'�A�e�m�e�X�3�'�����ӹ�ÇÁÅÇÓàëàÝÓÇÇÇÇÇÇÇÇÇÇ 0 P / ^ A 0 2 6 ; H , N < > ) s @ | w J ? X   ] ; M < 3 G \ v I � p G ` > p l M ^ % 8 . i d � V M > : i ( A [ ` s m X e + o 2 3 : > X Y & f > 4 I H Y M g x p ~ P    �  �  '  �  $  �  �  X  �  �  �  �  �  	  �  �  �  Z    ,  P  �  �  �  Z  �  t  �  b  �  �    N  �  t  /  �  r  �  �    �    !    \    �  ;  9  b  �  �  �  9  �  �  �    �  �  2  O  �  �    �  �  �  �  �    �  ^  �  �  �  x  r  �  m<e`B<�j;D��;ě��o�T����j�����C��o�<j�<j�D���C��u�����T����j�o�ě��e`B��/�,1�\)��㼼j��/��P�m�h���o��j�\)��P��/�C��@��\)��w��/�e`B�8Q����w�C��'aG���㽸Q콛��t��#�
��o�,1�H�9�H�9���
��C��}󶽸Q�T���Ƨ�m�h���`�����{������t���+�o�m�h���
�}󶽙�����T���
���㽶E���l�����BIB�[BJaBM�B�*B!FBWlBD�BPpB3/A�B�VB �B*B��BfzA��B!	A�]�BOB�
B&�nB)�B*!7B�4A��oBHB�jBr�A�EB �*B�9B.8"B1eB
f2B2�B�.B��B	�3B< B�_Bh�B��B�)B!rB� B
�B � B�B ��B�B�B
W&Bu�BGBp�B
rB�B6LBݟBXuB��B��B�B
��B�B-�0B��B&�B&�XB�*B!�2B"�:B :B
��B��B�
B�BTYB+B�BB��B@ZBB!BIFBE#B@�B>�B6 B@A��WB��B��B<�B��B�GB ?�BVeA�,�B�BF�B&��B=�B)ذB�{A�B�zBD�BJIA���B ��B�
B.<�B0İB
B>B>�B%DB?6B
=,B�B��BB�B��B�fB ��B��B�ZB �MB�B ��B1B�qB
�B��BI�B@�B
;B��BN�B�\B?0B@�B7�B��B
�PB��B.?B@�B&-{B&�KB��B!�RB#?�B@B@nB��B�B�B�BA�B�@F�;A���A���AƁAi�2A��yA��@��A،�AyrCA��OA烐A���A�)�A���A�AA�A�@�#PA��-A��sA,�sA��mB�Ax�bA�'�A�+�@��ZALzEA�*B�vA��A�|Ab� AY��A[0�AW�AA���A���A}n�A{��Amd�A�l�A̽A5@���A��!A���ȀDB	��@؆XA�>�A�A��A옎A��MA���A�?C�LTA5�A1��>��B@�ʸ@�wJB 9NA�_A�L�A
*A��A#�?A6[�A���@kf@@�JAO��A�V�Ax�/A	�A.�C�"�?�#�Aʅ6@DN�A��{A��RAƂ{Aj�cA���A͘�@�A� �Ay�A��A�/A�z�A���A��A�|*A�z�@� �A���A�߸A,�A���B3XAy�cA�XCA��5@�
AJ�YA�~zBB�A�F�A�t�Ad�LAU�A['RAV�>A��A�~�A�3A}AAl��A�t�Á	A5
�@���A�UA��A�qB	��@��zA�~�A��rA�A�HKA��CA�vQA�XLC�VHA6��A2�>�U�@��@�>�B :�AÂQA�}�A�A��>A#eA5�A��~@�@4`CAN�A� rAx��A7�A@�C��?�RRAʌ<               /         Q   &      *   *   +      	                     +                  	      '                                                          
   >   .                     %         -      3      6      %            N                     	      !                  ?      %   %   #         1               +               =      1                  !            !            -      !      !                        )                                 !      %            %                  %                     -                  5      %               )                              =      +                              !            %      !                                                                     #                              '                        NeF�N�eKN��(N%�8PRFGO8vO�>bOaO/6^N|�eOymP�Oc�^N�a�N�JO��O��MN I�OQ1O��O�P��N�0&O�@O�8�OV��NHN���Nh�UO�jNtըN�R�M��O�#�N�s�NN��OɗO#(�O���N
��Or.�N�AO ӰN�P�N=iO"�OB92N���Od$O��>O=�N�I�OtA�M�ojN���N���O<��N��OYBO$MN��O�UrN�E�N��N���O��/N��yO6�/NXg�Oa�dN��O���N?�N>BO)��N�9�N0x{O�,OY�dNJ�-  {  �  :  �  5  R  ~  	T  �  >  �    }  S  �  �  �  �  �  �  1  
    �  �  .  _  �    z  D  5  �  �  �  �  [  D  �  �  	  �  �    ~    @  /  z  V  	6  �    �  �  �  ]  
�  �  �  @  �  �  �  Y  �  �  �  �  �      /  �  �  D  �  �  #  �  U<���<�`B<�t�<e`B;D��;ě�;ě������D���o�e`B�T�����㼋C��o�#�
��t��49X�D���D���T���e`B��o��t����㼛�㼓t��������������9X��1��1��1������j�ě������ě����ͼ��ͽ\)���ͼ�/��/��`B��`B��h���ixս+���t��0 Ž#�
�0 Ž49X�49X�T���H�9��%�@��]/�D�����P�L�ͽaG��ixսP�`�]/��\)�P�`�T���ixս�o��o��C���C���O߽�E���v�)154,)'����������������������������������������56BBOQXOB>6655555555%)BOj����}t_RB)%�����������������������(*)
���������������������#/<GHMLG<7/+#��������������������9AHTaemoz|zqmaTH;:69�����������������|���
#'*..)#
������
#&./#
�������������������������������������������kz���������zmicccaak46BDKOPOB@>644444444SVamyzzwtqooma[VQPQS�����������������{|�
#*/;<><4/#
 ��0@Lc{������{U<0����������������������y���������������~yy���������������~xtu������		��������167BBDOTTOOMB6111111���(!��������������������������RT[amz~������zmaTSRR��������������������'),36?BHOPONFB76*)%'��
���������(.6COhu����k\UB6* !(pt����������togpppp����������������������������������������5N[g���tgaee[NB5**-5[`git~������tpjgbb[[������������������������������������������������������������������������������������������������������������������������#.+&#,/4<HUasz����zaUH</,vz���������������zvv������������������������������������������)-0133)�����46BLO[htuvsih[OB6,*4st����������tokissss��������������������()5BCDB53)((((((((((���

�����������|���������~{{z||||||	#(/<@>:951*#	\ht�������tlhd`\\\\ ))6341+)!

BFOS[htxwtrpjh[ZOMDB��������������������������������������35:BGNSZ[`[NB5523333stu�����������ztssss��������������������������# �������������������������#%0<FINRTRMI80# ;<AINRPLI<99;;;;;;;;()6BNTWXUOB6+)% ���������������������������
��������mnz�����zqnnmnonmmmm������������������������������znca`aanz�����������������������������������������(/9<HQUNH<9/)#v{�������������|xuuv���������������ֺֺҺֺ������������ֺֺֺֺֺֺֺ�������������
������
��������������¦±²¶²°¦�U�R�U�U�]�a�n�u�q�n�a�_�U�U�U�U�U�U�U�U�;�1���	�.�;�G�T�m������ÿÿ������m�;����������������������������������������ÇÁ�~Ëàìù��������������������ìàÇ�f�Y�X�Q�M�P�Y�f�r�����������������r�f�6�1�)�$�"�%�)�+�6�B�O�e�h�k�h�[�O�J�B�6�����������Ŀʿѿܿݿ޿ݿҿѿпĿ�������čā�w�m�d�h�r�tčĚĦĳĵĵĴıĨĦĚč������ĿĻĸĿ�������
��?�I�F�<��������Z�U�N�I�H�N�S�Z�g�����������������s�g�Z¿´²±²²¸¿����������������¿¿¿¿�Z�Z�W�Z�`�g�s���������������s�g�Z�Z�Z�Z�m�h�h�p�z�����������������������������m�����������������	��"�%� ��������������_�]�\�_�l�q�x�����x�l�b�_�_�_�_�_�_�_�_�N�A�+��(�5�A�N�g�s�������������s�g�Z�N����������������������#�,�1�4�*�����ݽԽнͽͽнԽݽݽ����������������������c�Z�N�A�+�0�N�g�������������������o�f�b�\�a�b�o�{ǅǈǍǋǈ�{�o�o�o�o�o�o�������m�j�������Ŀѿ����������Ŀ���	�
��	��"�/�;�H�N�T�Z�Z�T�H�;�/�"��	����������� �	��"�/�;�K�P�H�A�/�"��	�������������������ûûû����������������������������������������ʾ;ʾ���������������������������������������������ƳƨƎ�}�m�m�uƁƎƚƧƳ��������������ƳÓÍÇ�}�z�t�zÇÓÔßàéàÓÓÓÓÓÓ�H�A�;�/�*�"��"�/�;�@�H�T�W�a�c�a�a�T�H�G�F�;�9�.�-�.�;�?�G�K�H�G�G�G�G�G�G�G�G����پ̾Ⱦվվؾ�������	���%�"���������������	����"�#�"���	������������޾���������������������������������������������������������������v�n�~���������������������������������������ÿѿӿݿ����������ݿѿƿ��ֿͿĿ��������������ѿ������������ֿy�s�m�h�m�y���������y�y�y�y�y�y�y�y�y�y���������������)�8�B�H�B�;�5�)���àÜÓÓÓÖÛàìù����������ùùìàà�����(�.�2�4�5�4�2�(�"����
������~�x�l�j�_�]�Z�_�l�x�|�����������������B�<�5�,�5�B�N�W�[�_�[�N�B�B�B�B�B�B�B�B�U�J�H�F�E�B�H�H�J�P�U�a�j�j�f�g�c�`�Z�UìåàÞÔÌÇÅÇÓàìó����������ùì�$���������$�0�5�7�5�1�0�$�$�$�$�M�A�4�+�/�4�=�@�M�Y�^�f�q�r�|��r�f�Y�M�O�B�5�)�'�!�1�B�O�[�tāĊĄ�}�t�o�h�[�O�������������������������������������ŇŇņńŇŎŔŠŢŭűŴűŭŠŔŇŇŇŇ�/�#������#�0�@�I�Q�W�b�e�e�U�I�<�/������������������������������������������|�{�{��������������������������������¿´·¿��������������������¿¿¿¿¿¿E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eپ������(�4�A�A�A�>�4�-�(������� �����ݽݽ�������(�.�4�(����Ϲù����������ùϹչܹ�����������ܹϻܻڻػջܻ޻����ܻܻܻܻܻܻܻܻܻܻ��x�l�_�]�b�x���������̻лѻͻƻû������*�"����$�*�6�:�C�G�M�K�C�C�6�*�*�*�*�<�7�/�$�#�!�#�#�/�<�H�R�T�M�H�=�<�<�<�<�t�o�h�b�f�h�r�tāĂċčĖęčā�t�t�t�t��ּʼ¼Ƽʼռ������!�%�&���������������(�5�7�A�N�X�N�J�A�5�(�(��������������������������ĽнֽнνƽĽ��������(�4�;�>�4�(�����������s�g�f�g�p�����������������������������~�{�r�m�e�`�d�e�r�}�~�����������������~�����������������ɺ���"�!�����ֺɺ����������������ʾԾ׾׾׾ʾƾ�������������������ý�����������������������������ҿ����������Ŀѿֿٿڿܿڿѿ̿Ŀ��������������� �!�.�6�:�1�.�#�!�������:�9�:�D�G�S�`�l�`�U�S�G�:�:�:�:�:�:�:�:D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D����������3�@�N�M�@�=�3�'�����ÇÁÅÇÓàëàÝÓÇÇÇÇÇÇÇÇÇÇ   P - g J 2 2 $ # H " D 0 e & s 9 | w J ? X   N 5 L < 3 I U v I � p > ` B k l M ^  8 . i d � V N 8 3 i ( = [ ` s m > `  o ( 3 $ 5 J L $ X 8 4 J H T M g x p Y P    m  �  �  c  �    �  �  ~  �  �  �  �  �  �  �  i  Z    ,  P  �  �  s    �  t  �  �  C  �    N  �  �  /  �    �  �    �    !    \    �    X  8  �  �  �  9  �  �  �    <  D  2  �  �  	  �  e    ~  ]  �    �  ^  �  �  �  x  r  �  m  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�    K  `  n  x  {  x  j  Z  J  9  &    �  �  �    D  �  �  �  �  �  �  �  }  t  j  a  V  K  @  2  #      �  �  �  �  &  .  6  9  :  6  0  $    �  �  �  �  T  	  �  n    �  �  z  �  �  �  �  �  �  �  �  �  �  �  �  �  Z  �  �  p  -  �  �      *  4  2  *      
      �  �  �  c  1  �  �   �    7  H  P  Q  L  F  <  ,    �  �  �  �  l  @    �  e  B  ~  ~  v  h  S  8    �  �  �  �  �  �  r  V    �  �  (  �  #  n  �  �  	  	2  	I  	R  	K  	.  �  �  C  �  7  �  �  �  K  {  �       >  t  y  z  |  �  �  �  �  V    �  l    �  a  8  >  <  9  7  (      �  �  �  �  �  �  h  L  0     �   �   �  �  �  �  �  �  �  �  �  �  �  k  G    �  �  4  �  ?  �  �  �  �  �           �  �  �  �  �  d  )  �  �  I  �  �  �    4  O  b  s  |  }  x  l  U  0  �  �  g    �    z    �  �  �  �  �  �  �  )  Q  3    �  �  m  9    �  e  	  �  =  �  �  �  �  �  �  �  �  w  d  T  F  8  .  #    �  �  ~  /  �  �  �  �  �  �  �  ]  +  �  �  �  �  �  �  �  |  U  0  o  O  Z  o  }  �  �  �  j  N  .       �  �  �  �  y  +  �  B  �  �  �  �  �  �  �  }  z  w  �  �  �  �  �  �  �    "  6  �  �  �  �  �  |  j  W  @  )    �  �  �  �  �  f  B  �  �  �  �  �  �  �  �  �  m  \  Q  D  0    �  �  j    �  �  2  1  "    �  �  �  �  �  �  k  O  4         �  �  �  �    
  �  �  �  G  �  �  �    J    �  �  �  �  [  5  �  t   8         �  �  �  �  �  �  �  �  r  5  �  �  U    �  i    �  �  �  �  �  �  u  c  K  .    �  �  �  Z    �  �  P  9  �  �  �  �  �  �  y  [  :    �  �  �  z  H    �  �  M  8  -  -    
  �  �  �  �  u  B    �  �  �  ^  C  !  �  �  �  _  T  I  ?  3  !    �  �  �  �  �  �  �  v  b  2  �  �  �  �  �  w  l  k  b  D  &    �  �  �  �  `  ;     �   �   �   ~  �  �  �  �  �  �  �  �        	  �  �  �  �  d  �  L  �  3  h  v  z  u  j  O  /    �  �  q  6  �  �    �  7  �  �    %  .  :  ?  )    �  �  �  �  �  �  m  b  \  4    �  �  5  2  .  *  #      �  �  �  �  �  �  �  u  c  J  '  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  u  l  c  Z  Q  H  ?  6  �  �  �  �  �  �  �  �  �  j  ^  ^  :    �  �  U  )       [  x  �  �  �  �  �  �  {  m  Z  @    �  �  �  Y    �  e  �  �  �  �  �  �  �  �  �  �  w  \  A  &  
   �   �   �   �   m  T  X  [  Z  U  M  A  4  (      �  �  �  �  �  �  �  ~  w     +  =  &        
  �     �  �  �    @  �  �  �  �  1  �  �  t  �  �  �  p  [  B  &    �  �  �  i  9    �  �  /  �  �  �  �  �  �  z  e  F  /      �  �  �  �  N     �   }  	      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  �  �  �  �  �  �  �  �  �  �  �  �  e  /  �  �  5  �  =  >  �  �  �  �  �  �  �  d  9    �  �  x  N  '  �  �  �  x  H    �  �  �  �  �  �  �  �  �  o  U  9    �  �  �  x  I    ~  n  _  P  ;  "    �  �  �  R    �  �  h  @  &  <  c  �              �  �  �  �  �  �  h  B    �  �  �  g  9  @  2    �  �  �  H    �  �  c  `  _  E  -    �  �  �  �  /    �  �  �  w  F  (    �  �  �  �  C  �  �  &  �  -  �  k  r  y  x  v  o  e  X  I  4       �  �  �  ~  T  )  �  �  R  �  �  �      -  G  S  U  G  '  �  �  ~    �  �  )  �  	-  	6  	1  	"  	  �  �  r  9  �  �  x  +  �  _  �  d  �  �  �  �  �  �  �  �  �  �  �  n  \  J  7  %     �   �   �   �   �   �      �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  x  r  k  �  �  �  �  �  �  �  �  �  �  o  B    �  �  d    �  I  ^  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  y  t  n  j  f  b  ^  [  W  _  |  �  �  �  �  �  ]  X  R  M  G  B  >  9  3  ,  %      &  -  3  (    	  �  
�  
�  
�  
~  
R  
"  	�  	�  	V  �  �  9  �  q    �  �  �  �  &  |  �  �  �  �  �  �  �  �  �  }  ^  ;    �  �  {  9     �  �  �  �  �  �  �  �  �  �  �  �  s  �  |  L    �  �  F    d  m  �  �    #  8  @  8    �  �  O  \  @    �  �  %  $  �  �  �  �  �  �  �  �  ~  u  l  d  P  5    �  �  �  �  �  �  �  �  �  �  �  �  s  D    �  �  l  #  �  ]  �  U  �   �  �  �  �  �  �  �  �  l  P  2    �  �  �  �  {  V  0    �  -  p  �  �  �  �  �  �    Y  K  3    �  �  P  �  {  ?  �  �  �  �  �  �  �  �  u  W  6    �  �  �  X     �  v    �  �  �  �  �  �  �  �  �  d  7     �  �  4  �  �  5  �     W  �  �  �  �  �  �  �    S    �  �  n  ,  �  �  o  7  �  p  �  �  �  �    d  H  +    �  �  �  �  q  >  �  �  /  �    �  �  �  �  �  �  �  �  �  �  a  :    �  �  `  !  �  �  0  
!  
�  
�  
�        
�  
�  
�  
n  
"  	�  	B  �  �         �        �  �  �  �  �  �  �  �  �  �  ~  n  Z  E  @  A  A  /  ,      �  �  �  �  �  �        �  �  w  1  �  �  �  �  �  �  �  �  �  y  p  d  U  E  5  $      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  j  ]  D  *    �  �  �  �  _  5  	  �  �  u  <    �  �  n  A  �  �  p  I     �  �  �  ]  1    �  �  j  3  �  �  �  �  w  }  �  �  �  �  �  �  �  �  �  r  \  E  /    �  �  �  �  �  �  #    �  �  �  _  )  �  �  ~  =  �  �  �     �  �  1  y   �  *    �  �  �  �  �  �  �  b  3    �  {  +  �  �  ,  �  �  U  J  >  3  &      �  �  �  �  �  �  �  �    g  $  �  |