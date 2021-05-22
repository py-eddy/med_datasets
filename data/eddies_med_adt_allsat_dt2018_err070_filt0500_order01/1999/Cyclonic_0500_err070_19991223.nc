CDF       
      obs    F   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�|�hr�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�t{   max       P�Px       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��-   max       <��       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?5\(�   max       @F333333     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(�    max       @v������     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q�           �  6�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�5`           7`   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �n�   max       �ě�       8x   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�\-   max       B0=`       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�   max       B07A       :�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =�uj   max       C��       ;�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >?�"   max       C��`       <�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          {       =�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          O       ?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          E       @    
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�t{   max       P��,       A8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�����   max       ?ٛ=�K^       BP   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��-   max       %          Ch   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?=p��
>   max       @F(�\)     
�  D�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ҏ\(��    max       @v�(�\     
�  Op   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @Q�           �  Z`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�f`           Z�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         GM   max         GM       \   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��vȴ9X   max       ?ٙ�����     �  ]   <   f   x      &      $      %         @         {   (                  -         	   F            ,         	   *   9         	                                    
               	   
                           	            ,   P
>TP�PxP� JO%dSO���O��O���N<bO��`O@^�N�=�PUOOt˼N�BPZ[�OȢ�O0�N}}�O-�#O�^N�%�O��	N��Ok�O5;PTXLN�PjNj,�NH�PPG��OM��NP�O�P^4�P�TOIW�N,��N��3NTfjN��5O
̆Os�O�Y�NG`\N;�O�|1O=h}NdCN���N��NM"�NM~�N�-�OHgN�A�O(YNZ:�O5�fN�	�NҶ�N�N��4O(O�cN���N�z�M�t{Nk2�P�`N���<��<t�;��
��`B�t��T���e`B�e`B��o��C���t���t����㼛�㼬1��9X��j��j��j��j�ě����ͼ��ͼ���������/��`B��`B��`B�����+�t��t��t���P��P�''',1�,1�0 Ž49X�D���H�9�H�9�H�9�P�`�P�`�Y��]/�aG��e`B�e`B�ixսixս�o��������+��+��t���t�������
���T���T��-��-���)5BGJKE8�����
#<bx}nM<
���
���JT\v�����������mTJGJ������

������������

���������*6COWXUDC@6*)6BOS]hpri[XO63%7BNOWTOBA87777777777����������������������������������������X[bgtyztrg_[YRPRXXXX�#HZ���z}agaUH#
 ��?HTaimz}|zmeaTOGA;<?xz���������|zwxwxxxx)<HU_n~���znaU6#��'69=>=91�����#/<>HOJH<9/# #/3886/(# #/<HUabgaYUH</$# KUanz������znka`XUQKNO[^himhg[OMFDNNNNNN��
(/;8/%!
���������������������������)*6BJIMB;6)	Zht{���������vth[TOZ��������������������egqt�������tsga^eeee��������������������@ILUYbebWUMIDA@@@@@@��������8���������������������������SUabinxzznia_VUSSSS1;=HT^amryma[TH?;511%.Pb�������dL<0��������������������naXUNH</##/3<HUaqn�����������������������������������!)-1,)>BO[adc`\[VOJBA8>>>>��������������������������������������������������������������������������NN[^b[RNFFNNNNNNNNNNd��������������tlg\d��
#(&'))'
����y{�������{zvtuyyyyyy��������������������pt�����������tmppppp��������������������UUbnijnqnfbUSQUUUUUU)+5795-)����������������������
#,.*#
���������������������������aabnoz����znda`\aaaaeghntv����������tgde�������������|������,/79;<=AHJILOMH</+),ggstt������xtggggggg��������������������")3BNT[gmog[UNB.)("P\bdfpt���������tg[P��������������������� ��������������������������� ��������)5N[gt�~wg[NB5)$%)Z[aght|}����~th[UUZZ�Ŀ����������������ѿݿ����������ѿĽ��v�c�~�z�������f������f�P�4����Ľ�ƳƁ�h�C�
��,�CƁƧ������$�,�"�����Ƴ�Y�O�M�@�;�5�@�B�M�Y�f�r�z������{�r�f�Y��������Ϲ�������������ܹ���������׾ϾʾǾʾ׾��"�)�.�4�-���	�����l�K�<�@�L�S�l�|���������������������x�l�U�I�U�U�a�n�p�n�n�a�U�U�U�U�U�U�U�U�U�U�s�h�m�s�������������������������������s������ܻûȻлܻ���'�4�8�@�C�@�'�ƎƊƎƑƘƚƧưƳƽ������ƳƧƚƎƎƎƎ�Z�A�1�0�4�M�f����������ľ����������f�Z�(�$� �����(�5�A�]�g�k�l�g�Z�N�A�5�(���޿ݿؿݿ�����������������E�E�E�E�E�E�E�E�FF$FJFqF�F�F�FcF8FE�E��H�/����	��	��/�;�H�T�a�f�h�b�`�T�H�/�(�#�%�-�/�;�H�T�X�`�a�g�h�a�T�Q�H�;�/E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�¿·º´¿����������������������������¿�m�j�j�l�m�v�z���������������������~�z�m�r�r�q�r�}�����������������r�r�r�r�r�rE*E%E%E+E7E?ECEPEVEiEuEE�E�EuEiE\EPE7E*àÞÚÛàìù����������ùìàààààà�s�f�Z�X�Z�c�s�����������������������s��������x�s�m�s����������������������������}�x���������%�-�:�W�Z�S�!���ֺ����z�x�n�j�d�n�zÇÓàáààÙÓÇ�z�z�z�z�g�c�\�e�g�s���������x�s�g�g�g�g�g�g�g�g�ּʼʼɼʼӼּؼ�������ּּּּּ���������ìÓ�n�h�|Çñ���6�4����������
��
��#�/�<�H�U�a�e�a�W�H�7�/�#��(�"�'�(�5�9�A�G�N�Q�O�N�F�A�5�2�(�(�(�(���	���	����"�&�-�/�2�2�/�,�"���������s�A�-�	��(�N�s�����������������׿��������~�~�����������ɿҿ޿�����ۿĿ�àâìõ����������ùìáÕÓÊËÇÍÓà�;�:�;�A�G�H�T�W�V�T�M�H�;�;�;�;�;�;�;�;�������������
�������������������'�*�(�'���������������������������������������������������U�a�n�o�r�o�n�g�a�V�U�H�<�;�3�4�<�H�U�U�l�e�_�X�S�M�L�S�]�_�l�x�~�~�������x�l�l�������ּ߼����!�.�2�1�!������ּʼ�ŭťŠŠŔœŔŠŢŭŹŹ������Źŭŭŭŭ����������������������������������������ŔŊŁ�~�~ŇŢŹ������������ſŽŹŬŠŔ�û��������������������ûлܻ���ܻлü�
����'�4�?�@�A�@�4�'������������������$�0�1�1�0�$������������ּҼμʼƼɼʼּ�����������ּּּּ�߼�������������������㻷�������ûл׻ܻ�ܻڻһлû������������0�-�$�!� �#�$�.�0�=�A�I�J�L�I�I�=�<�0�0�������������������Ŀѿݿ���ݿѿпĿ��ݿտֿֿ׿ݿ������������ݿݿݿݿݿ��������������������
������
�������ؿ����������������ĿƿǿοѿѿѿĿ������������������������������������������ �#�0�:�<�>�<�2�0�#������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D����������������	�
�	����������������	�����������	��"�&�,�"�"���	�	�	�	�B�;�5�/�1�5�>�H�N�[�a�`�[�V�V�[�\�[�N�B������²�v�y¦²¿��������������ݽٽݽ�����������������!����!�.�:�?�G�I�G�<�:�8�.�+�!�!�!�!�S�M�S�]�`�e�l�y�}�y�l�`�S�S�S�S�S�S�S�S����������������������������������������}�t�u�z�y�uāĚĦ��������������ĺĦč�}���������������������ùιϹҹ׹ֹϹù���  R f ' � * ; E W u - : - c = G ' T P J $ ; )  \ P 3 [ G � R � G o < [ v 1 , 7 0 J o � H S N L e < n I ; 9 R G n ) @ d l V n K ? / � p % :  ^    c  Z  \  1  �  ]      �  �  �  �  �    s  �  �  c  �    �  �  �  �  �  �  p  T  �  �  R  �  �  �  �  �  j  �  3  Q  	  �    �  �  �  �    �  }    �  
  _  �  ~  �  *  u  �  �    �  �  7  �  A  ��`B��j��xսo�@��'L�ͼ����Y��,1��h���T��w�ě��n��}���C��0 Žt��#�
��O߽'<j�\)�ě��<j�+�o���P�H�9�\)�8Q콟�w��j�Y��,1�L�ͽT���T����7L�u����L�ͽY���O߽�C��]/��7L�y�#�u�e`B��C�������o��7L�u���㽟�w��{��C���t����
������� Ž��置{����/B5B%��B �oB#{�B� B0=`B4�Ba�B$�B  �B	<B��A�g�B LB�~B@KB B�B�DBԥB B)PB��B"�BG8B��B	ƗB�IB'�B� B1cB��A�\-B'�B+GyB�B�B	dB�BύB!`lB"M^B-L�BOjBx�BAB$|NB)`B�bB	SB��B'��BD�B[B�BdB��B
DCB ��B��B	�B��B��B
9�B�[B�zBBB��Bx�B��B=�B&@LB I�B#I�BC�B07AB<�BAXBǂB�/B�mB�A�uB �VB�BI�B	�B?�B��BʧB7�B>�B��B��B%WB�2B	��B��B&�2B�.B��B��A�B(?KB+8#B��B5�B��B?�B��B!>�B"=�B-�9B��B��B�2B$?�B)�B��B?�B�B'��BԥBM�BAzB�	BB0B
=�B ��B�{B	��B@�B��B
@nBD BʤBB�B�KB+B��Az��A0HB\�@�#�?	AX@�.A�?]A�q�@�K�B��A@M9A���A��[C��A���A�C��A���A�Z�@��C��A��AE�AHc@=��Aɓ~A��A	�A�o�A���A���A��>A�>>Aw~A��A�Y�@��u@� �A�uA�ro@���A�A���A���A�Ɲ@�
@�#0B	 �A��Aܹ@�|�B
z�Ax��A~N�A��dAx��A���A���C��AY[�A\uA���A�.�A0{AtAԁA!�A�;=�ujAz�SA3�lBA�@�%k>�(AX{@���A�~�A�~�@�n�B�A?|�A���A
�C��`A��A��OC�()A�t�A�F�@�aC���A�~�AE�BAJ��@>�3AɋA�� A{�Aт�A�w�A��uA�z�A�ޠAvVÂ\A���@���@���A��IAį8@� A�`A�P�A��A��S@��]@�g�B�A�A�@���B
G�Ax��AkA�39Aw�A���A�RC���AYA\�A�~�A�r�A0��A�yAA e�A���>?�"   <   f   x      '      %      &         @         {   )                  -         
   F            ,         
   +   9         
                                                   	                        	      
            ,      #   O   E      '      !               5         3   #                              5            E            ;   '                        %         !                                                      !               %         A   5                                    -                                 #            E            /   %                        %         !                                                                     %   O'��P��,Pd��NѷCO��O!��N�d�N<bOuX�O@^�N�=�O��O3��N�BP6ĔOe�N���NL�O-�#N渼N�%�NGEN��OG�5O5;O�[�N��Nj,�NH�PPG��OM��NP�N�qbP� O�b�OIW�N,��N��3NTfjN��5O
̆Os�O�Y�NG`\N;�O�|1OX�N1��N���N��NM"�NM~�N�-�OHgN�A�O(YNZ:�O5�fNh��NeQN�N��4O(O]#1Nv��N�z�M�t{Nk2�P�`N���  �  �  
�  �  a    �  0  #  I  �  �  x  Y  �  �  �  2  �  �  �  b  [  e  �  �    /  �  %  �  �  �      H  4  �  Q  :  9  �  %  8  �    �  [  D  �  �  9  �  �  M  >  P  �    0  �  �  C  r  _  "  �  �  u  �%   ��`B�C��e`B��9X��9X���e`B��9X��C���t���㼼j������+��`B������j���ͼě��P�`���ͼ�`B�����<j�o��`B��`B�����+��P�8Q�'�P��P�''',1�,1�0 Ž49X�D���H�9�P�`�L�ͽP�`�P�`�Y��]/�aG��e`B�e`B�ixսixս�o��+��hs��+��+��t��������P���
���T���T��-��- 	),4576-)� �
#IbowtbI0

���eq������������maVV[e�����


	�����������������������������"*-6CMOPNJC@6*"06BIOQUXYOB<60+*00007BNOWTOBA87777777777����������������������������������������X[bgtyztrg_[YRPRXXXX#<HUbjmgaU<4#
CHNTamzzywmkaTSLHFBCxz���������|zwxwxxxx!/<HUaz���znaUH:-$!���)0461)'���#/5<AC?</####/364/*##"#########/<HUabgaYUH</$# TUaanz�����znmcaZUTNO[^himhg[OMFDNNNNNN
!#$#
��������������������
)4>BGB@86)
	
Zht{���������vth[TOZ��������������������bgit�����tgdbbbbbbbb��������������������@ILUYbebWUMIDA@@@@@@��������8���������������������������SUabinxzznia_VUSSSS4;?HT\ampumaWTHC;744.16FVn{������{bI<1-.��������������������naXUNH</##/3<HUaqn�����������������������������������!)-1,)>BO[adc`\[VOJBA8>>>>��������������������������������������������������������������������������NN[^b[RNFFNNNNNNNNNNd��������������tlg\d�� 
##''%#
����w{�����{wuwwwwwwwwww��������������������pt�����������tmppppp��������������������UUbnijnqnfbUSQUUUUUU)+5795-)����������������������
#,.*#
���������������������������aabnoz����znda`\aaaaeghntv����������tgde�������������}������./;<=EHHJH<//,......ggstt������xtggggggg��������������������")3BNT[gmog[UNB.)("ot�����������tlijilo���������������������� ��������������������������� ��������)5N[gt�~wg[NB5)$%)Z[aght|}����~th[UUZZ�Ŀ������������Ŀȿѿݿ�������߿ݿѿĽ����~�������н���_�~����f�M�4���Ľ�Ɓ�h�I�2�0�B�O�uƎƧ�������
�������ƧƁ�Y�X�M�C�G�M�X�Y�f�o�r�~�����t�r�f�Y�Y�¹��ùϹֹܹ��������������ܹϹ¾��پ׾ҾϾ׾߾������ ����	����S�L�L�S�Z�_�l�x����������x�l�_�S�S�S�S�U�I�U�U�a�n�p�n�n�a�U�U�U�U�U�U�U�U�U�U���y�s�n�r�s����������������������������������ܻûȻлܻ���'�4�8�@�C�@�'�ƎƊƎƑƘƚƧưƳƽ������ƳƧƚƎƎƎƎ�Z�M�A�;�8�8�:�A�M�Z�f�l�q�����������s�Z�5�-�(�&�#��'�(�5�A�W�Z�g�h�g�Z�R�N�A�5���޿ݿؿݿ�����������������E�E�E�E�E�E�FFF$FJFiF{F�F�FvFcFJF1FE��H�;�1�%�"����"�/�;�H�R�T�Z�`�a�`�T�H�/�.�)�+�/�5�;�H�T�W�\�[�T�J�H�<�;�:�/�/E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�¿·º´¿����������������������������¿�z�u�m�m�o�p�z�~���������������������z�z�r�r�q�r�}�����������������r�r�r�r�r�rECE9E9ECEJEPERE\E^E\E[EPECECECECECECECECàÞÚÛàìù����������ùìàààààà�s�k�f�\�]�f�s�x���������������������s��������x�s�m�s������������������������������������ɺ�����������ֺɺ���Ç�}�z�o�u�zÇÓÛÜÖÓÇÇÇÇÇÇÇÇ�g�c�\�e�g�s���������x�s�g�g�g�g�g�g�g�g�ּʼʼɼʼӼּؼ�������ּּּּּ���������ìÓ�n�h�|Çñ���6�4����������
��
��#�/�<�H�U�a�e�a�W�H�7�/�#��(�"�'�(�5�9�A�G�N�Q�O�N�F�A�5�2�(�(�(�(���	���	����"�%�+�/�0�0�/�)�"�������s�Z�A�'���&�5�N�v���������������������������������������ſѿۿ���ӿĿ�àâìõ����������ùìáÕÓÊËÇÍÓà�;�:�;�A�G�H�T�W�V�T�M�H�;�;�;�;�;�;�;�;�������������
�������������������'�*�(�'���������������������������������������������������U�a�n�o�r�o�n�g�a�V�U�H�<�;�3�4�<�H�U�U�l�e�_�X�S�M�L�S�]�_�l�x�~�~�������x�l�l�������ּ߼����!�.�2�1�!������ּʼ�ŭťŠŠŔœŔŠŢŭŹŹ������Źŭŭŭŭ����������������������������������������ŔŊŁ�~�~ŇŢŹ������������ſŽŹŬŠŔ�û������������������ûлѻܻ߻���ܻлü����'�4�;�?�4�'���������������������$�0�1�1�0�$������������ּҼμʼƼɼʼּ�����������ּּּּ�߼�������������������㻷�������ûл׻ܻ�ܻڻһлû������������0�-�$�!� �#�$�.�0�=�A�I�J�L�I�I�=�<�0�0�������������������Ŀѿݿ���ݿѿпĿ��ݿտֿֿ׿ݿ������������ݿݿݿݿݿ��������������������
������
�������ؿ����������������ĿƿǿοѿѿѿĿ������������������������������������������"�#�0�8�<�=�<�1�0�#������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D����������������	�
�	����������������	�����������	��"�&�,�"�"���	�	�	�	�B�;�5�/�1�5�>�H�N�[�a�`�[�V�V�[�\�[�N�B¦²¿����������������¿¦����������	����������������������!����!�.�:�?�G�I�G�<�:�8�.�+�!�!�!�!�S�M�S�]�`�e�l�y�}�y�l�`�S�S�S�S�S�S�S�S����������������������������������������}�t�u�z�y�uāĚĦ��������������ĺĦč�}���������������������ùιϹҹ׹ֹϹù���  U d  B 0 ; E V u - E 0 c 5 @ ' e P ; $ & )  \ , & [ G � R � G X = [ v 1 , 7 0 J o � H S I K e < n I ; 9 R G n ) > : l V n ? 6 / � p % :  _  +  A  �  R  a    ]  4    �  �  �  �  (  �  �  V  �    �  T  �  �  �    �  �  p  T  �  �  ,    $  �  �  �  j  �  3  Q  	  �    �  J  ?  �    �  }    �  
  _  �  ~    t  u  �  �  �  �  �  7  �  A    GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  GM  �  y  �  7  p  �  �  �  �  �  �  �  �  �  G  �  �  �  �  �  Z  �  �  �  �  �  �  �  �  �  �  �  �  �  D  �  l  �  B   �  	K  	�  
A  
�  
�  
�  
�  
�  
�  
�  
b  
  	�  	,  m  �  �  �  p   �  �  �  �  �  �  �  �  �  �  �  T    �  �  H  �  �  ]    �  �  �  @  C  C  K  `  _  I    �  �  j  $  �  |    �  F  �  �  �  �  �  �             �  �  �  �  R    �  ~    �  A  F  K  T  k  �  �  �  �  �  �  �  �  h    �  p    �    0  :  B  H  H  G  A  6  (    �  �  �  �  c  .  �  �  �  D  �      #      �  �  �  �  �  �  g    �  a  �  �  ?  �  I  8  "  	  �  �  �  {  P  '    �  �  �  W  ;  N  �  �  Y  �  �  �  �  q  c  S  C  3  #    �  �  �  �  �  �  �  �  �  g  �  �  �  5  r  �  x  [  *  �  �    �  ~  |    W  5    V  f  p  v  x  v  p  e  W  C  )    �  �  �  S  	  �    Y  Y  N  C  8  0  5  :  ?  ;  *    	  �  �  �  �  �  n  J  &  �  l  �  �  �  [    �  2  �     4  N  
R  	B  %  �  u  !  �    D  s  �  �  �  �  �  �  �  b  5    �  �  E  �  |  �  L  �  �  �  �  �  �  �  �  �  �  �  �  l  F    �  �  j  +  �  $  *  /  1  1  2  ?  M  G  >  /      �  �  �  �  u  R  .  �  i  W  H  2    �  �  �  �    U  #  �  �  p  .  �  �  �  �  �  �  �  �  �  �  u  g  X  I  9  )      �  �  �  i  6  �  �  �  |  p  `  I  +    �  �  �  �  q  P    �  :  �  x  �  �  �  �  �  �  �  �    .  K  _  V  *  �  |  �  A  ^  N  [  V  M  A  4  *  !      �  �  �  �  �  f  =    �  �  >  A  d  d  `  Z  S  N  D  6  !    �  �  u  4  �  �  #  �  |  �  �  �  �  �  |  k  W  C  /    
  �  �  �  �  �  �  �  �  �  G  �  �  �  �  �  �  �  j  A    �  �  �    �    W  y  �  �            �  �  �  �  G  �  �  @  �  `  �  L   �  /  .  -  ,  *  (  &  #        �  �  �  �  �  �  |  a  G  �  �  �  �  �  �  �  �  �  s  c  R  A  0       �   �   �   �  %        �    	  �  �  �  ^    �  �  '  �  �  V    �  �  �  �  �  �  _  <    �  �  �  s  I     �  �  �  .  �  �  �  �  �  �  �  y  q  h  _  V  M  C  9  0  &      	   �   �  �  �  �  �  �  �  �  �  �  �  �  �  r  R  *    �  �  q  =    �  �  
      �  �  �  �  s  ;  �  �  g     �  /  �   �  �          �  �  �  �  a  -  �  �  E  �  t    n  �  �  H  <  )    �  �  �  �  �  �  �  x  f  R  ;    �  �  G  '  4  ,  $        �  �  �  �  �  �  �  �  �  s  [  D  -    �  �  �  f  L  3      �  �  �  �  �  �  �  �  r  ]  E  .  Q  I  @  7  )      �  �  �  �  J  �  �  K  �  �  ^  
   �  :  1  )  "        �  �  �  �  �  r  V  0  �  �  Z     �  9  5    �  �  �  ^  $  �  �  c  "  �  �  {  L  �  �  A  &  �  �  x  n  p  r  p  i  \  I  )    �  �  �  U    �  =  �  %          �  �  �  �  �  {  T  '  �  �  m  %  �  �  I  8  A  J  S  C  .      �  �  �  �  �  �  v  ^  E  &    �  �  w  n  e  \  R  H  >  3  &         �  �  �  �  �  q  N    �  �  �  �  �  �  �  ~  j  ^  g  [  5    �  �  �  G  �  N  Z  �  �  {  q  b  K  1    �  �  �  Y  !  �  �  O  �  2  Z  Z  Z  Z  Z  X  V  T  Q  N  K  H  F  D  C  A  C  E  H  K  D  D  >  3  '            9  j  �  �    K  �  �  %  p  �  �  �  �  �  �  r  ]  F  .    �  �  �  �  �  �  �  �  �  �  �  s  [  C  +       �  �  �  �  �  ^  >  *      �  �  9  ;  <  =  ?  @  A  C  D  E  A  8  /  &      
     �   �  �  �  �  �  �  w  Z  5    �  �  �  c  4    �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  \  >    �  �  �  X  )  >  O  M  E  =  4  *         �  �  �  �  �  �  }  `  C  #    �  >  9  3  *  !      �  �  �  �  l  C    �  �  �  p     �  P  E  9  .  "      �  �  �  �  �  �  �  �  �  �  �  y  n  �  �  �  w  k  \  L  :  &    �  �  �  y  J    �  �  �  �      �  �  �  �  �  �  ~  o  ]  L  9  %    �  �  �  `     �  �  �  �    #  -      �  �  �  z  M    �  m    �  ]  �  �  �  |  r  g  ]  R  H  >  2  %    
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  m  ]  L  ;  &    �  �  �  �  C  ,    �  �  �  �  �  �  �  �    k  6    �  �  b   �   �  @  E  c  l  q  r  p  i  [  @    �  �  �  l    �  4  �  h  L  U  ^  V  K  A  7  1  ,  )  '  &  %  #           �  �  "      
        �  �  �  �  �  �  �  �  �  �  �    G  �  �  �  �  �  �  �  �  �  �  �  �  �            %  -  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  p  u  p  k  f  _  M  3    �  �  A  �  �  -  �  P  �  a  �  /  �  �  n  T  E  N  M  L  F  ?  #  �  v    �  4  �  X  �  q