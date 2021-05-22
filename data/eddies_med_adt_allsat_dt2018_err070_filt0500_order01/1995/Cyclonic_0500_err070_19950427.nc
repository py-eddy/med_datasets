CDF       
      obs    [   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��+J     l  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       Pi��     l     effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���P   max       <�o     l   �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>h�\)   max       @F��\)     8  !�   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����P    max       @v�          8  0(   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @Q            �  >`   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�}        max       @�          l  ?   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ����   max       <t�     l  @�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�}   max       B4��     l  A�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B4�     l  C\   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��K   max       C���     l  D�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�HE   max       C��&     l  F4   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          2     l  G�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9     l  I   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7     l  Jx   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       PX�r     l  K�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�y=�b�   max       ?�-V�     l  MP   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���P   max       <�o     l  N�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>h�\)   max       @F��\)     8  P(   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min                  max       @v�          8  ^`   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @Q            �  l�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�}        max       @�e`         l  mP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @�   max         @�     l  n�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��PH�   max       ?�-V�     p  p(                  
   "      '            0                                          -   &                         ,         	      0                            !         1   #                                                #   !      	         
      .      
         	   !                     NN��NN�N&1N�"�M���N��6O���N��P
�OS�'Nx��N��RP$Z�O�>O�ɎNW��Oa� O�KNs3�O*b�O�5�OV]�OX\N�d�N�d�O��P	�Pa�Ob<O S�N��O'pgO+e�O.��N^��Pi��N���O5�O	U�N\�aO�S�N�mOK0 NH��O�QN���NCtbOC�O��'O��N���O��<O�p�P.�N�O*�N	��OA�tOu]gNSkO�u�Nz�O�^N�$N\?O ��N��0O��O��OJ�vO>FNN6<N�tQN^P
No �N�áN�O�03OqG�N�o�O�P�N���Nh�O���N�wBO N$��Nc�N���On�N2P <�o<�o<49X<#�
<#�
;�`B;��
;D��:�o��o��o�o��o�o�o�t��49X�49X�49X�D���D���T���T���T���e`B�u��o��o��o��o��C���C���t���t����㼣�
��1��j��j�ě����ͼ��ͼ�����`B��`B��h��h�������o�o�+�\)�\)�t���P��P��P�#�
�#�
�#�
�'',1�,1�0 Ž0 Ž49X�<j�@��@��H�9�H�9�H�9�P�`�]/�]/�aG��aG��m�h�m�h�m�h�m�h�q���u��������7L���P���Pjnz����znhjjjjjjjjjj��������������������)*-2565)& 
#&,0<0#
	�������������������������������������������
/-.,,$#
�������������������������������������������������������~}}|~������������GHUafebaUIHG>BGGGGGG-5BN[dk����t[N5)������������������)5;@EGDB)��pt�������tkpppppppp����#*(#
�������#/<UZXSOK@</)# ABDOW[b^[POB<8AAAAAA��������������������#*6COWeoiOC6*�����������
#(/8<HRMH/#
����������������������
#-/6/#
�����}������������������}<:=CHTmu~����zmaTE:<����5NXS;1)������������
�����������fhhkpt���������xtphf#*/90/#��������������������35=@<BBN[dgjkng[NB53)-6BOY[bhmka[ONB6))`ajmqrqma\Z\````````���#Wbnt~��y`I<0#��{����������������{v{������������������������ ������������Z[`chjpjh[YUUUZZZZZZ�1FJNKB6)������[[aght������~th]][[[����������������������������������������������� ��������������������������~��?BGOY[\hih_[OB??????�� 
#&#
������������!#!������
#4IfihkbUI<#
jmz}���{zmlmlljjjjjj#/HU]ad_^a\YU</")-6?IO[htywrqkh[O6/)0B[grhgmwtg[N5$��������������������/02<AIPUVXXWSI<;62//NNOX[gjg`\[NNNNNNNNN������������������������� �������������� ������������������������������{�������������������������������
#(#
������fgt������togffffffff��������������������������������������������������������������������������������������������������������������������������������������������{�����������znlhny{{����


�����������#-0<<IIID<0#v{����������{uopvvvvggrt���|���ytggggggg�����������������)043)���)*+01)
)<=5& ������������������������������������������#/<HUat|~zaUH</*# #������������������������������������������������������������������������������)*5?BNUQNHB75)z��������������zwuvz���������������������H�F�A�>�H�U�X�Z�Y�U�H�H�H�H�H�H�H�H�H�H�U�M�L�O�R�U�a�d�c�a�`�\�U�U�U�U�U�U�U�U�I�D�H�I�V�b�m�o�o�r�o�b�V�N�I�I�I�I�I�I�����������������������ü¼������������������������ʾ˾Ͼʾ�������������������������������������#�#������������8�(�#�#�'�5�A�N�Z�g�s�t�y�v�q�g�b�Z�N�8������������������������������������������Ʈƶ������� � �#�$� �&�$������������������������/�6�;�=�6�*��������U�U�O�T�U�a�a�d�i�m�n�z�|�z�n�a�U�U�U�U�������(�)�/�5�5�5�(��������Ŀ������}�w�{�����ѿ��������տѿĿ��������������Ŀѿݿ������������ݿĿ��s�i�Z�M�P�U�f�s����������������������s���������������������������������������àÓËËÓÚàãìù������������ýùìà�����������������������	���	����������U�T�U�Y�a�g�n�zÃ�z�z�r�n�a�U�U�U�U�U�U�U�I�H�>�<�7�4�<�H�U�a�l�n�q�u�y�n�k�a�U�.�"�	���������	�"�.�8�:�=�?�?�F�;�.����������������������!�#���������¿µ·¿������������������������������¿ìëââìù��������ùõìììììììì�����%�)�/�6�;�B�G�M�H�B�?�7�6�)���4�-�4�;�A�I�P�S�Q�X�f����������s�Z�M�4ƎƁ�u�h�c�]�^�hƁƧ����������
����ƧƎ������������������������������������������{�����������������ĿƿʿĿ�����������	�����ܾؾؾ����	����"�&�"���нǽ˽н׽ݽ������ݽнннннннл��ֻܻл��лһܻ�������������s�q�f�Z�M�A�A�:�9�A�F�M�Z�f�s�{�}�}�u�s�Z�T�L�K�L�T�V�`�h�m�y���������}�z�r�m�Z�O�K�O�Q�[�h�t�y�|�t�h�[�O�O�O�O�O�O�O�O���������}�Z�K�B�Z�g������������� �������(���
��������������(�;�:�5�(ǭǥǡǚǔǈ��{�s�l�n�o�{ǈǔǟǡǭǭǭ��	�� �	��"�/�:�;�H�T�X�T�O�H�;�/�"��@�=�@�L�R�Y�e�r�r�t�r�e�Y�L�@�@�@�@�@�@�g�U�W�d�����������������������������s�g�������������������������������������������������������)�6�B�H�C�G�B�6�-��¦§°­¦ŔœŇōŔŗŠŭųŹ��ŻżźŹųŭŠŔŔùõìàÓÇ��}ÇÓàìôù��������ùù�T�R�H�>�;�:�;�E�H�J�T�Z�^�V�T�T�T�T�T�T���{�z�q�m�j�m�z�����������������������������º����������������ɺֺ�������ֺ��!��������-�:�F�_�q�q�s�h�k�j�_�F�-�!����������������������������������������������������������������������������Ź�ܹӹù����������ùϹܹ�����������������ŻŻ�������C�O�]�\�_�_�N�6���ߺ��� ����������&�'�+�'�%����������������(�4�A�N�O�M�A�4�(������������������������������������������F�;�6�9�:�@�F�S�_�d�l�x�}�������l�_�S�F���ߺ�����!�-�6�:�=�F�I�F�:�!�����	����������	�����	�	�	�	�	�	�	�	�H�C�E�H�T�Z�a�h�m�z�}�������������m�T�H�m�h�g�l�m�n�z�{�|�z�m�m�m�m�m�m�m�m�m�mŠŜŔŐŜŭŶŹ��������������������ŭŠ�л̻ǻû��ûлػܻ�����������ܻлп	�� ���	�������	�	�	�	�	�	�	�	�	� �����������	��"�"�%�"�"���	�	�������������
���
�������������������������������)�6�B�U�T�V�N�B�5�)���������������������ʼּ��ݼ߼��߼ּʼ��@�5�'�$�)�4�@�M�Y�f�r�������q�f�Y�M�@āĀ�t�h�[�Z�[�h�tčĚĦĬĪĦĜĚĐčā�@�7�>�@�E�L�Y�]�^�Y�M�L�@�@�@�@�@�@�@�@�����������������ĿȿſǿſƿĿ����������Ľý��������ĽннڽӽнĽĽĽĽĽĽĽĽ���������������������������������������������#�'�4�@�E�G�@�<�4�'�����ĳĲĦģğĦĭĳĶĿ��Ŀĸĳĳĳĳĳĳĳ��f�Y�M�@�6�.�.�+�4�@�Y�������������������������{�������������Ľнݽ��нĽ��������������������������������������ý�����������)�B�^�p�`�O�6������������
��#�/�.�#��
��������������������������������������������������������FF	FFF#F5FJFVFoF|F�F~FyFmF[FAF1F#FF����������������������������������������
���'�,�3�D�L�c�L�@�3�'����/�-�$�/�<�H�H�Q�H�<�/�/�/�/�/�/�/�/�/�/�.�)�!���!�.�6�:�<�=�:�.�.�.�.�.�.�.�.�z�y�n�n�k�a�_�Z�a�n�r�zÀÇÇÍËÇ�z�z¦£¦¿��������������������¿²¦�������������
�����
���������������� a � r R 8 4  d 9 Q _ C E + ; 3 9 C ] 6 I 2 l : l v W > B < 7 - , L O ^ x % j b M , B W M v [ G P ; x W 1 O 7 7 k . H g W w . E x 9 8 O N 8 . b s R V > u 1 8 0 x G A k 5 w ] U { $ \   �  �  �  �    	    �  �  �  �  �    �  C  j  �  9  �  o  �  �  r  �  $  �  #  �  P  `  �  x  |  �  �  �  T  �  n  �  k    �  a  C  -  �  (  �  A  �  [  >  >  �  A  S  �    n  �  a    �  �  _  �  7  �  �  �  �  L  i  �  �  G  �  �  �  �  �  �  "  �  �  1  �    �  I<t�<t�<o;D��<o�o��`B���
���T����`B�o�T���ě��ě��T���o�+��j�C��o��P���ͼ�j���ͽt��u�]/��j����9X�H�9�����ͼě���%������w�o��h�����P�ixս+�,1�,1�o�t��q����%�t��@����O߽@��aG��#�
�u�q���<j�u�49X�e`B�H�9�<j�ixսP�`��C���o���
���
�P�`�m�h�e`B�]/�u�e`B���ͽ�t������ Ž�o��7L��^5��\)��C���t���hs��hs��^5���
BTDB[�B��B$�\B4��BH�B��B�VB� B ��B��B��B��B{�B�B
/�Bh�Bx�B��B!-�B0fB��B%BsB(�B �XA�3�B��B��B��B��B*�By=Bu�A���B&�B)��B]�A�}Bf�B�+B'�B�B��B��B �^B�XB�B#*@B&	A��B��B��B�@B!��B&�%B��B^�B��B�B��B�7B��B$\PB
�B��B!B�zB+p<B"Bx#B 3�B��B#��B%�`B)+�B	ޘBRB@B��B�Ba�B��B��B��B+B
��BܤB:�BBbB�cB�*B�qB$��B4�B@�BB�DB��B �	B��B��B	6�B#gB��B
:BC�BBXB��B �B0HNB�{BAhB�SB�]B!R�A��B/�B��B��B�iB��Bb�B�tA��LB&w1B*CB��A��Bt�B?�BA�B��B��B�B ��B#�B�B#�SB&>A���B75B<�B�@B!�B&��B�{B@JB��B�fBNBB�B�xB#�7B	��B�B?�B��B*�eB!�|B��B�BB�wB#�	B%��B)@'B
=�B?�B?�B�B��B�#B��B:�B�)B:�B
�=B��B�wB
D�B?A��bAŏ	B��@��AO�>A13�A���A�e�Bv�A�*;A��`A��Ay�^Az�ZAE��A�A̘�A�4SA�9:A��0A\�~A��OA�.A�&9A�DnAB٢BRA��bAs�AX=OA*��@�G�A?4bAj��A�$>A�q�A��B��A�7�?�fA��hA���A�8,A�.�A��A�� A�WA� �@1��@se�A�r�A��>��KA��f?y�A6��A��e@�@diAZX�A�
�A��sA���@�:�A[4�AZ��A�*�A�	�@��1@أA�Q�?�=�Aw��A&�hA!�t@�$�A��@�/A$[�@�x�A��A�(4A廼C���B��?��]AÇ�A�nA��hA��A�!$A�x�A�w�BI�@�D�AO�A0�pA��A�B+B?�A��A�vA��A}&Ax
AE��A��$Ȃ�A��VA�g{A�|�A[8�A�~�A��A�t�Aք*AD�uB�A��YAt*WAWA+�@��8A>��Aj�zA�z�A�y\A��B:A��Q?�_�A��]A�vAՎHA���A���A�|	A���A�A�@3iK@s�'A��yA���>�HEA���?thA7�-A�V�@��@T�AZ�6A�x�A��~A�|�@�n�AZ�AY�]A�MA���@��@�2�A܇6?�XKAw�A'�A!@ͧKA��@ݴ�A' @�W\A��A�WtA�| C��&B��?~��A�A�AǍA��A���                  
   "   	   (            1                                          -   '            !            ,         
      1                            !         2   #                              	         	         #   "      
         
      /               
   "      	                                          )            /                        !               %   +   3                        9               %                        '   %            +                     #                        #                           !         '         %                                                            %                                          +   %                        7               #                        '               !                     #                        #                                    '         %                     N/԰NN�N&1Np	EM���N��6O�uN��O���OS�'Nx��N��RO�RO�>O�ɎNW��Oa� O2�NO��O	O@�BO8ɧN��XN�d�N�d�Om&HP��Px�Ob<N��-N��O��N��O.��N^��PX�rN���O�O	U�N\�aO��oN�mOwNH��N�[�N���NCtbOC�O��'O-M@N���Oh�`OT�0O�D�N�O*�N	��OA�tOe�
NSkO�u�Nz�Oe�N�$N\?O ��N��0OIS�O��OB�]N���NN6<N�tQN^P
No �N��jN�O��OqG�N�o�O�P�N���Nh�O���N�wBO N$��Nc�N���On�N2P   �  �  �     �  �  {  Q    �  ?  �    �  Y  .    �  �  >  �  
  Y  �  z  <  �  �  �  �    �  6  �  �  #  �  �  �    [    6  �  �    �  1  �  �  l  �    J  3  �  G  �  �  �  !  s  E    �  �  �  p  X  �  �  �  �  �  {  x  w  �  �  A  �  �  A  �  �  �  A  �  �  �  7<u<�o<49X<t�<#�
;�`B��`B;D���D����o��o�o�e`B�o�o�t��49X��o�D���u��C��u�e`B�T���e`B��t���C�������o��t���C���1��1��t����㼴9X��1���ͼ�j�ě��o���ͽ+��`B����h��h�����<j�o�+�,1�49X�\)�t���P��P���#�
�#�
�#�
�,1�',1�49X�0 Ž<j�49X�@��m�h�@��H�9�H�9�H�9�Y��]/��%�aG��aG��m�h�m�h�m�h�m�h�q���u��������7L���P���Plnz����znillllllllll��������������������)*-2565)& 
#%+/#
������������������������������������������
"#%$#
�����������������������������������������������������������~}}|~������������GHUafebaUIHG>BGGGGGG3BNR[^dt~���vg[N?--3������������������)5;@EGDB)��pt�������tkpppppppp����#*(#
�������##/<HPOKHD</&#8BOV[a\[YOB=88888888��������������������'*6CNOUYZZKC6*�����������#'/7<HPJH</#����������������������
#-/6/#
�������������������������:=HTmtz�����zmaTF:<:����)5BKJ2.#����������
�����������rt���������~tlkmrrrr#*/90/#��������������������?BLN[^fggkg_[NBBCB??)-6BOY[bhmka[ONB6))`ajmqrqma\Z\````````���
#0bnz��w^I<0���{����������������{v{������������������������ ������������Z[`chjpjh[YUUUZZZZZZ��,:@CB<6)����[[aght������~th]][[[�����������������������������������������������������������������������������~��?BGOY[\hih_[OB??????�� 
#&#
������������!#!������#<INUWWUOI50-#jmz}���{zmlmlljjjjjj#/HQU\\^`[XU</#1<CO[hiturnnlh[XOB61"5[`]][a`b`[NB5)$""��������������������/02<AIPUVXXWSI<;62//NNOX[gjg`\[NNNNNNNNN���������������������������������������� ������������������������������{����������������������������������
#(#
������fgt������togffffffff��������������������������������������������������������������������������������������������������������������������������������������������{�����������znlhny{{����


�����������#-0<<IIID<0#w{���������{xswwwwwwggrt���|���ytggggggg�����
������������)043)���)*+01)
)<=5& ������������������������������������������#/<HUat|~zaUH</*# #������������������������������������������������������������������������������)*5?BNUQNHB75)z��������������zwuvz���������������������H�G�B�@�H�U�W�Y�X�U�H�H�H�H�H�H�H�H�H�H�U�M�L�O�R�U�a�d�c�a�`�\�U�U�U�U�U�U�U�U�I�D�H�I�V�b�m�o�o�r�o�b�V�N�I�I�I�I�I�I�������������������¼������������������������������ʾ˾Ͼʾ�������������������������������������#�#������������A�8�5�/�1�5�A�E�N�Z�_�g�n�n�h�g�Z�N�A�A������������������������������������������Ƽ��������������������������������������������/�6�;�=�6�*��������U�U�O�T�U�a�a�d�i�m�n�z�|�z�n�a�U�U�U�U�������(�)�/�5�5�5�(������������������������ѿݿ��������Ŀ������������������Ŀѿݿ������������ݿĿ��s�i�Z�M�P�U�f�s����������������������s���������������������������������������àÓËËÓÚàãìù������������ýùìà���������������������������������������a�U�Z�a�h�n�z�|�z�y�q�n�a�a�a�a�a�a�a�a�U�L�H�?�<�9�:�<�H�U�[�a�g�n�q�r�n�h�a�U��	�����������	��"�.�1�4�5�0�.�"�������������������������!�������¿¶¸¿������������������������������¿ìëââìù��������ùõìììììììì�����%�)�/�6�;�B�G�M�H�B�?�7�6�)���M�M�S�V�T�W�^�f�s����������������s�Z�MƁ�u�d�^�_�h�uƎƳ��������������ƧƎƁ���������������������������	��������������{�����������������ĿƿʿĿ������������ھ۾����	������	��������нǽ˽н׽ݽ������ݽнннннннл��ܻٻѻػܻ���������������M�G�A�?�>�A�M�Z�[�f�m�s�v�w�s�f�Z�W�M�M�Z�T�L�K�L�T�V�`�h�m�y���������}�z�r�m�Z�O�K�O�Q�[�h�t�y�|�t�h�[�O�O�O�O�O�O�O�O������������v�Y�Z�g���������������������(���
��������������(�;�:�5�(ǈǆ�{�z�t�p�s�{ǈǔǜǡǪǫǣǡǖǔǈǈ��	�� �	��"�/�:�;�H�T�X�T�O�H�;�/�"��@�=�@�L�R�Y�e�r�r�t�r�e�Y�L�@�@�@�@�@�@�r�g�]�^�k�����������������������������r�����������������������������������������������������)�4�6�<�<�6�)�������¦§°­¦ŠŚŔŌŐŔŜŠŭŶŹŶůŭŠŠŠŠŠŠùõìàÓÇ��}ÇÓàìôù��������ùù�T�R�H�>�;�:�;�E�H�J�T�Z�^�V�T�T�T�T�T�T���{�z�q�m�j�m�z�����������������������������º����������������ɺֺ�������ֺ������!�+�-�:�>�F�O�F�E�F�L�F�:�-�!�����������������������������������������������������������������������������Ź�Ϲù��������¹ùϹܹ���������������������������*�6�C�O�T�T�O�H�6�*����ߺ��� ����������&�'�+�'�%����������������(�4�A�N�O�M�A�4�(������������������������������������������F�;�6�9�:�@�F�S�_�d�l�x�}�������l�_�S�F��ߺ�����!�-�3�:�<�F�H�F�:�-�!�����	����������	�����	�	�	�	�	�	�	�	�H�C�E�H�T�Z�a�h�m�z�}�������������m�T�H�m�h�g�l�m�n�z�{�|�z�m�m�m�m�m�m�m�m�m�mŠŘŔŒŭŸŹ��������������������ŹŭŠ�л̻ǻû��ûлػܻ�����������ܻлп	�� ���	�������	�	�	�	�	�	�	�	�	������������	���"�#�"����	�	�������������
���
������������������������ ��	���)�+�5�B�O�M�B�5�,�)��������������������ʼּ��ݼ߼��߼ּʼ��@�6�'�%�)�4�@�M�Y�f�r�������r�o�f�Y�@�t�s�h�a�`�h�tāčĚĝěĚčČā�t�t�t�t�@�7�>�@�E�L�Y�]�^�Y�M�L�@�@�@�@�@�@�@�@�����������������ĿȿſǿſƿĿ����������Ľý��������ĽннڽӽнĽĽĽĽĽĽĽĽ��������������������������������������������'�+�4�@�@�D�@�8�4�'������ĳĲĦģğĦĭĳĶĿ��Ŀĸĳĳĳĳĳĳĳ�Y�@�<�:�7�6�<�@�M�Y�r�����������r�f�Y�����������{�������������Ľнݽ��нĽ��������������������������������������ý�����������)�B�^�p�`�O�6������������
��#�/�.�#��
��������������������������������������������������������FF	FFF#F5FJFVFoF|F�F~FyFmF[FAF1F#FF����������������������������������������
���'�,�3�D�L�c�L�@�3�'����/�-�$�/�<�H�H�Q�H�<�/�/�/�/�/�/�/�/�/�/�.�)�!���!�.�6�:�<�=�:�.�.�.�.�.�.�.�.�z�y�n�n�k�a�_�Z�a�n�r�zÀÇÇÍËÇ�z�z¦£¦¿��������������������¿²¦�������������
�����
���������������� ` � r A 8 4  d % Q _ C R + ; 3 9 ' Z 3 2 . e : l G W ; B $ 7 $ ' L O [ x + j b G , = W . v [ G P F x C 1 W 7 7 k . H g W w + E x 2 8 9 N 7 6 b s R V : u # 8 0 x G A k 5 w ] U { $ \    h  �  �  �    	  )  �    �  �  �  D  �  C  j  �  ~  �  9  �  �  ;  �  $      �  P  �  �  "  �  �  �  �  T  K  n  �  �    6  a  �  -  �  (  �  �  �    �    �  A  S  �  �  n  �  a  �  �  �  !  �  �  �  �  �  �  L  i  �  �  G  1  �  �  �  �  �  "  �  �  1  �    �  I  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  g  @    �  �  �  x  c  M  7  !  	  �  �  �  �  r  B  "        �  �  �  n  [  I  7  $      �  �  �  �  �  �  �  �  p  ]  I  5  �  �            �  �  �  �  �  �  �  t  V  9    �  �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �  �  �  �  �  t  f  X  G  3      �  �  �  �  �  m  Q  >  +  �  �  #  H  \  n  x  {  w  d  L  -    �  �  @  �  }  $  �  Q  K  D  =  6  0  .  +  #    
  �  �  �  �  �  h  8  �  �  �  v  �  �           	  �  �  �  _  '  �  �  +  �  �   �  �  �  �  �  {  i  W  D  1      �  �  �  �  c  :    �  �  ?  L  Z  g  g  d  a  ]  X  T  V  `  j  l  ^  P  @  &    �  �  �  �  �  �  }  r  e  Y  L  =  ,      �  �  �  [     �  �  �  �  �    �  �  �  �  b  S  j  N  "  �  �  -  �  �  �  �  �  �  �    y  s  _  I  1    �  �  �  z  E    �  �  �  Y  Q  G  9  '    �  �  �  �  �  �  {  \  6    �  |  <   �  .  '          �  �  �  �  �  �  �  �  �  �  �  �  �  �      �  �  �  �  �  �  v  M  %  �  �  �  }  I    �  �  �  }  |  |  �  �  v  ^  C  $    �  �  ~  B  �  �  i  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  V  �  s    !  1  :  <  6  (      �  �  �  �  f  J  "  �  �  &  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  R    �  ~    �      �  �  �  �  �  |  e  a  M  -    �  6  �  k  �  �  G  R  Y  Y  Q  F  7  "  
  �  �  �  �  �  W    �  z  (  �  �  �  �  �  �  y  h  U  @  (    �  �  �  �  �  _  -  �  �  z  i  W  C  ,    �  �  �  �  y  c  W  N  B  )    �  �  �  �  "  (  <  6  )      �  �  �  �  T  %  �  �  |  )  �  �  �  �  �  �  r  d  O  <  '    �  �  R  �  �  K  �  F  y  p  !  i  z  �  �  �  �  �  k  G  (  	  �  �  a    �  -  �  h  �  �  �  u  d  T  A  -      �  �  �  �  �  �  {  t  p  l  �  �  �  �  �  �  �  �  �  �  �  �  �  Z  %  �  �  F   �   �      	                    �  �  �  �  �  �  �  �  8  o  �  �  �  o  Z  @    �  �  �  c    �  i  �  V  �    (  ,  /  3  5  5  3  )      �  �  �  �  o  R  8  -    �  �  �  �  �  �  �  �  �  �  k  S  ?  D  J  I  @  6  '      �  �  �  �  x  g  W  F  =  ?  A  C  A  =  9  5  /  )  #        �  �  ~  I  *  �  �  �  {  T  .    �  �  �  W  �  "  �  �  �  �  �  z  q  h  `  \  X  T  I  9  )            �  �  �  �  �  �  �  �  �  �  {  ]  6    �  �  \    �  �  �  �  o  d  Y  L  ?  0       �  �  �  �  �  u  :  �  �  y    �  �  �  �  �  �  �  �  �  �  �  �  �  w  i  ]  S  H  =    /  M  Y  M  ;  $  �  �  �  i    �  l  �  t  �  �  <  �                  �  �  �  �  �  W  '  �  �  {  =     h  �  �  "  5  1  #    �  �  �  n  <    �  �  _  	  �  U  �  �  �  �  �  �  �  x  g  T  A  /    
  �  �  �  �  �  �  �  �  �  �  �  �  }  a  B     �  �  �  �  c  4  �  �    N    �  �  �  �  �  g  5    �  �  K    �  \    �    >  �  �  �  �  �  �  �  �  �  �  v  i  [  N  B  :  3  +  #      1  )  !         �  �  �  �  �  �  �  �  �  �  s  S  3    �  �  �  x  q  h  Z  G  @  9  /  &      �  �  }  3  �    �  �  �  �  �  �  �  �  �  �  �  �  }  K    �  t  �  E  �  l  f  _  Y  R  I  :  +      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  q  a  L  1    �  �  �  P    �  �  R  �  �  	        	  �  �  �    G    �  �  @  �    �  �  �  �    7  B  J  A  "  �  �  �  �  F  �  �  _  �  �  4  ^  3  3  0  *  "          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  ^  I  0    �  �  �  l  *  G  ?  6  -  %          �  �  �  �  �  �  �  k  J  (    �  �  �  �  �  z  W  1    �  �  �  r  @    �  �  P    �  �  �  �  �  �  �  �  �  x  ^  9  �  �  c    �  �  G  �  c  �  �  o  ]  F  -    �  �  �  �  �  �  l  9    �  �  �  m  !        �  �  �  �  �  ~  i  R  7    �  �  y  ;  �  J  s  i  ^  T  I  @  ;  6  0  +  ,  3  ;  B  I  R  [  d  m  v  =  C  C  =  2  !    �  �  �  �  �  �  c  @    �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  �        $  /  <  I  �  ~  u  l  d  Y  K  <  -      �  �  �  �  �  �  s  ]  F  �  �  �  �  �  �  �  �  v  f  T  >  %  	  �  �  �  t  f  �  �  z  e  M  3    �  �  �  �  �  \  3  �  �  d  5     �   �    T  j  f  S  9    �  �  �  �  �  �  s  ?  	  �  �  k  R  X  @  %  	  �  �  �  �  �  w  X  0  �  �  �  �  l  9  �  �  �  �  �  �  l  X  A    �  �  ?  �  �  #  �  =  �    v  1  �  Y  w  �  �  �  �  �  �    Z  3  	  �  �  i    �  u    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  O  +    �  �  �  �  �  }  _  ?    �  �  S  �  �  �  �  y  k  [  J  :    �  �  �  �  r  E     �   �   z  {  m  `  S  E  4  #    �  �  �  �  �  }  Z  6     �   �   �  p  r  u  v  w  t  o  a  P  :  !    �  �  �  �  n  K  %     w  c  P  <  )       �   �   �   �   �   �   �   �   �   �   �   �   {  <  O  �  �  �  �  �  �  w  K    �  �  H  �  �    &  %  �  �  �  �  s  `  Q  D  4     	  �  �  �  �  �  f  .  �  �  v  A  7  ,    �  �  �  e  B  !     �  �  �  ~  ^  ?  "  1  F  �  �  �  {  ^  =    �  �  i    �  v  C  
  �  �  �  ~    �  �  �  �  �  �  �  �  �    v  m  c  ^  b  g  r  �  �  �  A  D  H  O  V  Y  [  V  N  C  6  (      �  �  �  `  (  �  �  �  �  �  �  \    �  �  D  �  �  {  *  �  E  �    �  �  �  �    u  p  o  i  a  Z  R  F  7  %    �  �  �  �  p  Q  �  �  �  �  �  w  ]  C  '    �  �  �  �  �  �  �  �  �  �  A  4  (    
  �  �  �  �  �  �  x  X  9    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  r  i  a  Y  R  J  C  �  �  �  �  �  �  w  d  P  =  )      �  �  �  �  �  �  k  �  �  �  t  K  !  �  �  �  o  ?    �  �  �  e  `  �  �    7  5  4  2  -  )  $      	  �  �  �  �  �  �  \  @  #  