CDF       
      obs    =   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��l�C��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�/   max       P��m      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��F   max       <�/      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @E�G�z�     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�    max       @v��G�{     	�  *   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @6         max       @Q�           |  3�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��`          �  4   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��-   max       <�C�      �  5   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�(�   max       B.5      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B.>�      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�a�   max       C���      �  7�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >���   max       C���      �  8�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          n      �  9�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?      �  :�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ?      �  ;�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�/   max       P��m      �  <�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��Q�   max       ?���C,�      �  =�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��F   max       <ě�      �  >�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?W
=p��   max       @E�G�z�     	�  ?�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�(�X    max       @v��G�{     	�  I   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @4�        max       @Q�           |  R�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�          �  S   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?�   max         ?�      �  T   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�hr� Ĝ   max       ?���C,�       T�   
            +      	                     7            	         	                           M      m         &      #            
      
      	               &                  	               N撆N��O��N���P1�zO��YO-� O�gO�K�O��Nm��ND��N+�2Pw�O(g�O*�Ov�N��'M��O2��N�$�N��O-nN�}M�/N�6O�OJNZ><On�IP_+BN��P��mO�N~��O�O�OC�PʕN�AwNb�$N�ԶN�bO6�0O�(N��N���N�O&�O?O �OwaFN6�)N�.�O��N3eOAxkM�:NLN��)O;�O11N��J<�/;ě��o�ě���`B�#�
�49X�49X�D���T����C���1��9X�ě����ͼ�������������`B�����o�o�\)��P���,1�,1�49X�49X�8Q�8Q�@��L�ͽP�`�Y��Y��]/�m�h�m�h�q���q���q���y�#��%��%�����7L��O߽�O߽�O߽�hs��hs������
��{��{��9X��9X��G���F5<AIUblmjbaUIF<95555xz��������{zvxxxxxxUanz�����������nUOPU�������������������������������������xw����)2::2%	���������������������������������������������/;Hamz}rma^TLGC;/��������������������JO[hjjh`[XOCJJJJJJJJ��������������������������������������������#bn{����{<0
��������������������������� ):)������ABJOU[hqohghh^[WOC>Aghjty�������tplhgggg����������������������� $#�������eht�������������tnhe����������������������������������������>BIOQ[][VOKB>>>>>>>>���

�������������269=?=661..022222222����������������������������������������KN[t���������tgVNCDK�����������������}����������������#/HVapl\a���zU</##`imwz��~~}zmea^Z[`jmz�����zmkijjjjjjjj��������������������#0<HU_adfeaUHD</$#7DHUnu����zwkaUH<717������������������������������������������������������������LO[hlstuuuthd[ZTONLLqt�����������trqrttq�������
��������q{����������{zwtqoqq�������������������������������������������&'!�������{��������������zvx{.5>BN[gt����tg\SNH8.��������������������#(/35/'#���������������������������������������������%&$! �������������������������##$,,'$#########�������������������������	����������CHJPYan�������znULHCABFN[^_^[YNB>?AAAAAA�����������������ʼּּܼټּͼʼ������������������������������������������������L�D�D�I�`�g�d�f�s���������������i�g�Z�L�U�P�L�U�a�b�n�z�{�{�{�s�n�b�U�U�U�U�U�U������A�Z�|����������������f�M�4��[�N�B�5�.�.�5�B�N�[�gª²»¦�Ŀ����ÿĿοѿݿ�������������ݿѿ�àÜÔÓÏÍÍÓàìðøùþ��ûùìàà���������(�5�N�X�q�s���s�Z�N�6��ݿѿϿ������Ŀпۿ��������������ÓÊÇËÓàåìðìåàÓÓÓÓÓÓÓÓ�;�:�.�*�.�;�G�T�Y�`�T�G�;�;�;�;�;�;�;�;��{��������������������������������������x�`�Y�W�e�s�������������������z�y�~ÇÎÓÜàäìðùúúóìàÓÇ�z���޹ӹԹ�ܹܹ�������������Ϲιù��������ùϹڹܹ����������ܹ��<�6�/�(�/�8�<�H�Q�U�X�a�Z�U�H�=�<�<�<�<�U�S�H�D�F�H�U�X�\�V�U�U�U�U�U�U�U�U�U�U�Z�X�U�Y�g�s�y���������������������s�g�Z�������������������ûл׻лƻ����������������������������������������������������s�k�g�Z�P�P�X�Z�g�s�������������������s�����������������ûŻû���������������������������������������������������"��"�.�;�G�T�^�T�G�;�.�"�"�"�"�"�"�"�"���������"�H�R�T�U�T�I�?�:�/�"��z�m�m�a�m�m�z���������|�z�z�z�z�z�z�z�z����������	���������	�����I��������������
�0�I�nŀ�~�r�p�|�q�b�I��������������	��������������������E�E�E�E�E�E�E�E�E�FFF=FbFXF[FTF=E�E�E���ƳƭƧƢƧƳ����������� ��������������ĳĪĬĳļĿ����������ĿĳĳĳĳĳĳĳĳĿľıĦĘĖďĚĳ��������� ����������Ŀ�Z�M�A�<�8�9�A�E�M�Z�f�u�s�y�s�l�l�f�e�Z���������{����������Ǿ׾�������ʾ��r�p�n�g�e�b�e�n�r�~�������������~�t�r�r�����ݿܿҿݿ������
������������������������������������Ŀ̿Ŀ�����������������������������������������������������������������	��"�-�/�/�'�"��	�������ֺغ��������� �!�"� ��������S�I�I�S�S�_�l�x�}�������������x�l�_�S�S�N�H�B�5�)�!����(�*�5�B�J�N�[�^�_�[�N��	����$�0�1�0�$�����������f�e�f�j�r�r�����������������������r�f�����������'�@�M�Y�\�U�O�M�@�4�'�ā�~āĄĊċČĐėĚĦĬ��ĿĳĦĞĚčāčć�~�t�c�]�c�h�tāčĚĤĩĤĢģĠĚčD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DƼ}�x��������������¼ʼռʼƼ����������}��ؼ������!�.�2�8�.�(�!��������[�X�O�C�O�T�[�e�h�l�h�^�[�[�[�[�[�[�[�[�����������Ľнݽ�����������ݽнĽ���D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�Dƽ����������ĽнֽнĽ����������������������������	���'�(�*�(�������������������������!�/�:�I�I�I�G�:�.�!���������z�m�i�e�d�g�m�z�������������������a�_�U�H�I�U�a�n�w�zÄÁ�z�n�a�a�a�a�a�a A 2 v < V � @ Z L = 1 c P ^ S � : T g I h , V E H ~ T 4 ] B j X b 1 @ > K e e L Q M W H m � 6 K � ^ h l ` 3 N � q S 2 o T      �  v  �  _  s  |  c  �  �  q  V  ^  �  �  h    �  Q  �  Y  �  �  /  
  �  %  e  )  �  9  �  �  �  �  �  �  "  y  �  �  �  ]  '    �  x  �  �  @  o  8  p  A  �  p  w  �  �    �<�C���o��C��D���L�ͽt����
��h�'�`B���
������h�����0 Ž49X�D���\)�t��#�
��w��P�<j��w�#�
�49X�Y��H�9��%���ٽD����-�u�]/��9X��hs��9X�y�#��%��%��O߽��w��O߽�hs��t���\)��Q콬1���"ѽ���罰 Ž��
��j��vɽ�^5���������$�B'�B�/B�zB6�B �wB�UB��BJJA�(�B�oB+�B -tB uAB&k=Bw�B^MB6�B��B ��BM�B��BX�B��B��B$?B�=B� BW�B	�B�B/Be�A�JjA�K�B8�BB��B ;PB*8�B+�8Bn�B5nB#T�B)�Bc�BРB�>B)�WB��Bs[B�JB+��B.5BI}B=�B�B%=B��B�nB:B\bB'9�BB@[B;6B @1B��B(&BưA���B��B�B ?�B }�B&/BK�B?�B=�B�B �B@�B��B9�B�B�NB#�UB�B%_BAYB	��B;:B�KB�$A��"A�	�B?�BU�Be|B ��B*(�B+�BS�B;@B#AjB)9EB�B?�BʖB)��B	RB�lB�-B+��B.>�B?�B?�B@PB%@B��BC{B)�B?}@��$A��JA�R�A�ȷA@"bA�_IA}�,A��A���A�A��Ad��@�Y"A��Aʥ�?0�>�a�A�I�A�s�A��2@���A��A�#E@�98@�K�Ab��A��FA�JAY��A� XA���C���B~.A�+A�:&A>i�AR�)@KXA�AuqA�5$A��@\L@��A�B	wy@��@�E~A���Aݠ�C�@��A�A�I}A)2,C�hA$�eA�]�A��A���A���@�o�AІaA���A��AB�UA�~�A �A�}fA���A}H�A���Ac�u@�:oA���AˍE?#>���AěPA��A�@�<A���A�@@��@�GAcR�A���A��AYU�A�pzA���C���B��A��MA�]A<��AR�@ ��A�!�As��A���A�>�@`�1@��iA���B	B&@鍥@�{@A�HxA�t�C� @��PAP�A��jA*�UC�A#�A��6A�A�}�A�}7               ,      
                     8            
   	      
                           N      n         &      $                        
               '                  	                        '      /   %         !               9                                                5      ?         !      )                                                                                 '      -                           7                                                -      ?                                                                                       N�DZNh�0O��N���P,4OOq�O-� O �Oo�BO�d�Nm��ND��N+�2Pr��O(g�O�BN:��N��'M��O2��N�N�N��O-nN�}M�/N�6O�OJNZ><O�&P)&�N��P��mO�N~��O�~RO#�~Ox�zN�AwNb�$N�ԶN��Ob�N��\Nܾ�N���N�O�O{O �OI�N6�)N���O��N3eOAxkM�:NLN��)O;�O}�N��J  �  �  A  \  K    m  �  5    %  �  �  :  >  �  �  %  4  �  �  U  (  �  �  �  �  �  b  	�  U  �  �  -  �  �  U    �  P    .  �  '  j  �  J    �  	  �  �  �      N  c     �  Y  u<ě�;�o�o�ě��o�e`B�49X�T����t��u��C���1��9X���ͼ��ͼ�/�t�������`B�����o�o�\)��P���,1�,1�L�ͽe`B�8Q�8Q�@��L�ͽ]/�aG�����]/�m�h�m�h�u�y�#�y�#�}󶽁%��%��+��\)��O߽�����O߽�t���hs������
��{��{��9X��9X��S���F:<IRUbfiebXUOI<<::::yz|���������~zyyyyyyUanz�����������nUOPU�������������������������������������yx���)-66."���������������������������������������������BHTamz{ytmgbaTLHFA>B��������������������JO[hjjh`[XOCJJJJJJJJ�������������������������������������������
#bn{�����{<0
��������������������������(�������JOR[hkih_[OOJJJJJJJJghjty�������tplhgggg����������������������� $#�������htz�������������tqhh����������������������������������������>BIOQ[][VOKB>>>>>>>>���

�������������269=?=661..022222222����������������������������������������T[agt��������tkg[VTT����������������������������������#/HVapl\a���zU</##`imwz��~~}zmea^Z[`jmz�����zmkijjjjjjjj��������������������!&3<HU]acdbaUIH</*#!9<BHUanomjfaUHF@<;:9������������������������������������������������������������LO[hkqstrhe[[UOOLLLLttwx������������tsrt�������

���������s{�����������{xurpss������������������������������������������%& �������{}��������������~{y{.5>BN[gt����tg\SNH8.��������������������#(/35/'#���������������������������������������������%&$! �������������������������##$,,'$#########�������������������������	����������QU[anz�������znaULQQABFN[^_^[YNB>?AAAAAA�����������������ʼϼּؼּԼʼɼ������������������������������������������������L�D�D�I�`�g�d�f�s���������������i�g�Z�L�U�P�L�U�a�b�n�z�{�{�{�s�n�b�U�U�U�U�U�U������A�Z�~����������������f�M�4��[�N�E�5�1�2�5�B�N�[�g�w�t�[�Ŀ����ÿĿοѿݿ�������������ݿѿ�àÞÖÓÑÏÎÓàìîõùüÿúùìàà�������(�5�A�Q�Z�g�j�o�Z�R�N�5�(���ݿѿɿ����ƿӿݿ���������������ÓÊÇËÓàåìðìåàÓÓÓÓÓÓÓÓ�;�:�.�*�.�;�G�T�Y�`�T�G�;�;�;�;�;�;�;�;��{��������������������������������������y�a�[�[�h�s�������������������z�y�~ÇÎÓÜàäìðùúúóìàÓÇ�z���߹Թչ�ܹܹ���������������Ϲǹù����ùϹѹܹ�ܹ۹ϹϹϹϹϹϹϹ��<�6�/�(�/�8�<�H�Q�U�X�a�Z�U�H�=�<�<�<�<�U�S�H�D�F�H�U�X�\�V�U�U�U�U�U�U�U�U�U�U�Z�X�U�Y�g�s�y���������������������s�g�Z�������������������ûлӻûû����������������������������������������������������s�k�g�Z�P�P�X�Z�g�s�������������������s�����������������ûŻû���������������������������������������������������"��"�.�;�G�T�^�T�G�;�.�"�"�"�"�"�"�"�"���������"�H�R�T�U�T�I�?�:�/�"��z�m�m�a�m�m�z���������|�z�z�z�z�z�z�z�z���������������	��������	�����I�0��
���������,�I�n�u�u�l�j�p�n�b�I��������������	��������������������E�E�E�E�E�E�E�E�E�FFF=FbFXF[FTF=E�E�E���ƳƭƧƢƧƳ����������� ��������������ĳĪĬĳļĿ����������Ŀĳĳĳĳĳĳĳĳ��ĿĳĦĚęĖĚĦĳ�������������������̾Z�M�A�=�:�:�A�I�M�Z�f�i�o�s�s�i�i�f�^�Z�ʾ������������ʾ׾����
��	�����׾ʺr�p�n�g�e�b�e�n�r�~�������������~�t�r�r�����ݿܿҿݿ������
������������������������������������Ŀ̿Ŀ����������������������������������������������������	���������������	��"�#�+�-�-�$�"��	����������������������������S�K�J�S�V�_�l�x�{�������������x�l�_�S�S�N�H�B�5�)�!����(�*�5�B�J�N�[�^�_�[�N��	����$�0�1�0�$�����������f�f�j�r�r������������������������r�f�'�������!�'�4�@�M�W�Q�M�M�@�:�4�'ā�~āĄĊċČĐėĚĦĬ��ĿĳĦĞĚčā�t�t�h�h�b�h�k�tāčđĚĝġĚęčā�t�tD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DƼ�����y��������������ü�����������������ؼ������!�.�2�8�.�(�!��������[�X�O�C�O�T�[�e�h�l�h�^�[�[�[�[�[�[�[�[�����������Ľнݽ�����������ݽнĽ���D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�Dƽ����������ĽнֽнĽ����������������������������	���'�(�*�(�������������������������!�/�:�I�I�I�G�:�.�!���z�t�m�j�f�e�d�h�m�z�����������������z�z�a�_�U�H�I�U�a�n�w�zÄÁ�z�n�a�a�a�a�a�a J * v < V p @ ^ > 6 1 c P Y S ~ > T g I k , V E H ~ T 4 J A j X b 1 < > C e e L S ; : ? m � 5 < � ) h P ` 3 N � q S 2 e T    �  q  v  �  T  m  |  >  �  _  q  V  ^  �  �  �  Z  �  Q  �  (  �  �  /  
  �  %  e  E    9  �  �  �  �  x  �  "  y  �  �  N  �      �  X  8  �    o  �  p  A  �  p  w  �  �  �  �  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  �  �  �  �  �  �  �  �  �  �  �  �  ~  d  J  /    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  U  "  �  �  �  W     �  A  =  6  *        �  �  �  �  h  B       �  �  m  H  7  \  T  K  B  6  *      �  �  �  �  �  �  k  Q  7      �  B  H  <  *    �  �  �  �  |  P    �  �  �  G  �  �  �  }               �  �  �  �  �  �  B  �  �  C  �  V  �    m  k  i  k  m  g  _  R  C  1      �  �  �  �  N     �   �  �  �  �  �  �  �  �  �  t  2  �  �  o  9    �  m  �  b   �  �  �    (  5  3  &    �  �  �  �  a  8    �  V  �  5  �  �          �  �  �  �  �  �  �  j  Q  7    �  �  l    %  !            	    �  �  �  �  �  �  �  �  �  �  �  �  y  s  l  d  X  M  A  3  "    �  �  �  �  �  e  C      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  w  u  t  6  7    �  �  P    �  k    �    5  �  �  T    �      >  3  7    �  �  �  �  S  2    �  �  s  9    �    x  �  g  �  {  R    }  `  9  	  �  �  V    �  �  �  #  �  7   �  ^  o  �  �  �  �  �  �  �  �  �  �  �  ^    �    ~  �  ^  %          �  �  �  �  �  �  �  �  �  �  l  R  ;  @  E  4  .  )  "          	    �  �  �  �  �  �  a  ?    �  �  �  �  �  �  �  �  w  ]  @  '    �  �  �  �  x  <   �   �  y  �  �  �  �  �  z  q  g  [  L  =  -      �  �  �  �  l  U  R  N  K  F  @  :  3  *        �  �  �  �  �  k  S  ;  (        �  �  �  �  �  ]  6    �  �  �  �  �  k  :  �  �  �  �  }  q  e  \  R  H  ?  /      �  �  �  �  �  y  `  �  �  �  �  �  �  �  �  �  �  �  �  �  �    %  E  e  �  �  �  �  u  e  P  9  "    �  �  �  �  �  �  |  c  I  '     �  �  z  s  j  ^  O  :  $    �  �  �  z  T  ,  �  �  �  G  
  �  �  �  �  �  �  �  o  W  :    �  �  �  �  �  `  >     �  �    8  O  Y  ^  b  a  X  E  -    �  �  y  ;  �  �  r     �  	-  	�  	�  	�  	�  	e  	?  �  �  s    �  (    �  �    �    U  S  Q  N  L  J  G  L  U  ^  f  o  x  ~  �  �  �  �  �  �  �  �  {  D  �  �    �  w    �  Q  
�  	�  	0  �  A  `  =  �  �  �  |  o  b  S  B  +      �  �  W    �  �  7  �  �  0  -  &          �  �  �  �  �  �  �  �  �  �  �  y  o  e  �  �  �  �  o  W  =  $  
  �  �  �  z  L    �  u    �  �  �  �  �  �  �  �  }  f  J  '     �  �  �  ^     �  �  9  +  �  �  �  �  �  �  A  Q  >    �  �  p  '  �  �  R  Z  X      �  �  �  �  �  �  �  �  �  �  �  �  p  Z  A  )    �  �  �  �  �  �  �  �  �  �  �  �  m  Z  G  3       �  �  �  z  P  A  2  #      �  �  �  �  �  �  �  �  �  �  s  `  L  8          �  �  �  �  �  V  -  
  �  �  �  �  ]  1  E  `    
  .    �  �  �  �  Z  1  	  �  �  �  O    �  �  �  A  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  b  O  >  G  R  &  &  &            �  �  �  �  �  _  4  �  �  �  I     �  j  P  6  !    �  �  �  �  �  �  �  �  �  �  �  f  J  ,    �  d  9      �  �  �  �  �  �  �  �  �  �  �  |  o  b  V  6  I  C  :  +      �  �    A     �  x  -  �  �  f  3  �  �  �  �  �  �  �  �  �  �  �  _  <    �  �  �  y  2  �  [  �  �  �  �  �  �  �  �    _  9    �  �  �  S  "  �  �  �  �  q  S  y  �  	  �  �  �  �  :  �  �  -  �  b  �  �  �  6  �  �  z  U  *  �  �  �  k  4  �  �  }  ?  �  �  z  5  �  �  �  �  �  �  �  �  �  �  l  V  ?  '    �  �  �  �  X    �  �  �  �  �  j  K  )    �  �  �  h  0  �  �  Y    �  N   �    �  �  �  �  �  �  �    i  R  ;      �  �  �  �  b  B    �  �  �  �  �  �  �  w  Y  7    �  �  �  �  b  *   �   �  N  '     �  �  {  <  �  �  g    �  {  *  �  �  4  �  �  7  c  l  u  ~  o  ]  J  6  !    �  �  �  �  �  y  ^  F  .           �  �  �  �  �  w  J  	  �  �  >  �  �  f    �  �  �  �  �  �  q  \  F  0        �  �  �  �  �  /  �  �  R    Y  ;    �  �  �  �  �  q  J  !  �  �  C  �  �  J  �  �  u  f  T  >  #    �  �  �  t  K     �  �  �  X    �  �  C